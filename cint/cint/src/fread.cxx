/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file fread.c
 ************************************************************************
 * Description:
 *  Utility to read source file
 ************************************************************************
 * Copyright(c) 1995~2003  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"

int G__fgetvarname(G__FastAllocString& string, size_t offset, const char *endmark);
int G__fgetname_template(G__FastAllocString& string, size_t offset, const char *endmark);
int G__fgetstream_newtemplate(G__FastAllocString& string, size_t offset, const char *endmark);
int G__fgetstream_template(G__FastAllocString& string, size_t offset, const char *endmark);
int G__getstream_template(const char* source, int* isrc,G__FastAllocString&  string, size_t offset, const char* endmark);
int G__fgetname(G__FastAllocString& string, size_t offset, const char *endmark);
int G__getname(const char* source, int* isrc, char* string, const char* endmark);
static size_t G__getfullpath(G__FastAllocString &string, char* pbegin, size_t i);
int G__fdumpstream(G__FastAllocString& string, size_t offset, const char *endmark);
int G__fgetstream(G__FastAllocString& string, size_t offset, const char *endmark);
void G__fgetstream_peek(char* string, int nchars);
int G__fgetstream_new(G__FastAllocString& string, size_t offset, const char *endmark);
int G__fgetstream_spaces(G__FastAllocString& string, size_t offset, const char *endmark);
int G__getstream(const char* source, int* isrc, char* string, const char* endmark);
static int G__isstoragekeyword(const char* buf);

extern "C" {

#ifdef G__MULTIBYTE
int G__CodingSystem(int c);
#endif // G__MULTIBYTE

int G__fgetspace();
int G__fgetspace_peek();
int G__fignorestream(const char* endmark);
int G__ignorestream(const char* source, int* isrc, const char* endmark);
void G__fignoreline();
void G__fignoreline_peek();
int G__fgetline(char* string);
void G__fsetcomment(G__comment_info* pcomment);
void G__set_eolcallback(void* eolcallback);
int G__fgetc();
} // extern "C"

#ifdef G__MULTIBYTE
//______________________________________________________________________________
extern "C"
int G__CodingSystem(int c)
{
   c &= 0x7f;
   switch (G__lang) {
      case G__UNKNOWNCODING:
         if (0x1f < c && c < 0x60) {
            /* assuming there is no half-sized kana chars, this code does not
             * exist in S-JIS, set EUC flag and return 0 */
            G__lang = G__EUC;
            return(0);
         }
         return(0); /* assuming S-JIS but not sure yet */
      case G__EUC:
         return(0);
      case G__SJIS:
         if (c <= 0x1f || (0x60 <= c && c <= 0x7c)) return(1);
         else                                return(0);
      case G__ONEBYTE:
         return(0);
   }
   return(1);
}
#endif // G__MULTIBYTE

//______________________________________________________________________________
static int G__isstoragekeyword(const char* buf)
{
   if (!buf) return(0);
   if (strcmp(buf, "const") == 0 ||
         strcmp(buf, "unsigned") == 0 ||
         strcmp(buf, "signed") == 0 ||
         strcmp(buf, "int") == 0 ||
         strcmp(buf, "long") == 0 ||
         strcmp(buf, "short") == 0
#ifndef G__OLDIMPLEMENTATION1855
         || strcmp(buf, "char") == 0
#endif
#ifndef G__OLDIMPLEMENTATION1859
         || strcmp(buf, "double") == 0
         || strcmp(buf, "float") == 0
#endif
         || strcmp(buf, "volatile") == 0
         || strcmp(buf, "register") == 0
         || (G__iscpp && strcmp(buf, "typename") == 0)
      ) {
      return(1);
   }
   else {
      return(0);
   }
}

//______________________________________________________________________________
static bool G__IsIdentifier(int c) {
   // Check for character that is valid for an identifier.
   // If start is true, digits are not allowed
   return isalnum(c) || c == '_';
}

//______________________________________________________________________________
int G__fgetname_template(G__FastAllocString& string, size_t offset, const char *endmark)
{
   //  char *string       : string until the endmark appears
   //  char *endmark      : specify endmark characters
   // 
   //    read one non-space char string upto next space char or endmark
   //   char.
   // 
   //  1) skip space char until non space char appears
   //  2) Store non-space char to char *string. If space char is surrounded by
   //    quotation, it is stored.
   //  3) if space char or one of endmark char which is not surrounded by
   //    quotation appears, stop reading and return the last char.
   // 
   // 
   //    '     azAZ09*&^%/     '
   //     ----------------^        return(' ');
   // 
   //  if ";" is given as end mark
   //    '     azAZ09*&^%/;  '
   //     ----------------^        return(';');
   // 
   size_t i = offset, l;
   int c, prev;
   short single_quote = 0, double_quote = 0, flag = 0, spaceflag, ignoreflag;
   int nest = 0;
   int tmpltnest = 0;
   char *pp = string + offset;
   int pflag = 0;
   int start_line = G__ifile.line_number;

   spaceflag = 0;

   do {
      ignoreflag = 0;
      c = G__fgetc() ;

      if ((single_quote == 0) && (double_quote == 0) && nest == 0) {
         l = 0;
         while ((prev = endmark[l++]) != '\0') {
            if (c == prev) {
               flag = 1;
               ignoreflag = 1;
            }
         }
      }
backtoreadtemplate:
      switch (c) {
         case ' ':
         case '\t':
         case '\n':
         case '\r':
         case '\f':
            if ((single_quote == 0) && (double_quote == 0)) {
               string.Set(i, 0);  /* temporarily close the string */
               if (tmpltnest) {
                  if (G__isstoragekeyword(pp)) {
                     if (G__iscpp && strcmp("typename", pp) == 0) {
                        i -= 8;
                        c = ' ';
                        ignoreflag = 1;
                     }
                     else {
                        pp = string + i + 1;
                        c = ' ';
                     }
                     break;
                  }
                  else if ('*' == string[i-1]) {
                     pflag = 1;
                  }
               }
               if (strlen(pp) < 8 && strncmp(pp, "typename", 8) == 0 && pp != string + offset) {
                  i -= 8;
               }
               ignoreflag = 1;
               if (spaceflag != 0 && 0 == nest) flag = 1;
            }
            break;
         case '"':
            if (single_quote == 0) {
               spaceflag = 1;
               double_quote ^= 1;
            }
            break;
         case '\'':
            if (double_quote == 0) {
               spaceflag = 1;
               single_quote ^= 1;
            }
            break;

         case '<':
            if ((single_quote == 0) && (double_quote == 0) &&
                  strncmp(pp, "operator", 8) != 0
               ) {
               int lnest = 0;
               ++nest;
               string.Set(i, 0);
               pp = string + i;
               while (pp > string && (pp[-1] != '<' || lnest)
                      && pp[-1] != ',' && pp[-1] != ' ') {
                  switch (pp[-1]) {
                     case '>':
                        ++lnest;
                        break;
                     case '<':
                        --lnest;
                        break;
                  }
                  --pp;
               }
               if (G__defined_templateclass(pp)) ++tmpltnest;
               pp = string + i + 1;
            }
            spaceflag = 1;
            break;

         case '(':
            if ((single_quote == 0) && (double_quote == 0)) {
               pp = string + i + 1;
               ++nest;
            }
            spaceflag = 1;
            break;

         case '>':
            if ((single_quote == 0) && (double_quote == 0) &&
                  strncmp(string + offset, "operator", 8) != 0) {
               --nest;
               if (tmpltnest) --tmpltnest;
               if (nest < 0) {
                  string.Set(i, 0);
                  return(c);
               }
               else if (i && '>' == string[i-1]) {
                  /* A<A<int> > */
                  string.Set(i++, ' ');
               }
               else if (i > 2 && isspace(string[i-1]) && '>' != string[i-2]) {
                  --i;
               }
            }
            spaceflag = 1;
            break;
         case ')':
            if ((single_quote == 0) && (double_quote == 0)) {
               --nest;
               if (nest < 0) {
                  string.Set(i, 0);
                  return(c);
               }
            }
            spaceflag = 1;
            break;
         case '/':
            if ((single_quote == 0) && (double_quote == 0)) {
               /* comment */
               string.Set(i++, c);

               c = G__fgetc();
               switch (c) {
                  case '*':
                     G__skip_comment();
                     --i;
                     ignoreflag = 1;
                     break;
                  case '/':
                     G__fignoreline();
                     --i;
                     ignoreflag = 1;
                     break;
                  default:
                     fseek(G__ifile.fp, -1, SEEK_CUR);
                     if (G__dispsource) G__disp_mask = 1;
                     spaceflag = 1;
                     ignoreflag = 1;
                     break;
               }
            }

            break;

         case '#':
            if (single_quote == 0 && double_quote == 0 && (i == offset || string[i-1] != '$')) {
               G__pp_command();
               ignoreflag = 1;
#ifdef G__TEMPLATECLASS
               c = ' ';
#endif
            }
            break;

         case EOF:
            G__fprinterr(G__serr, "Error: Missing one of '%s' expected at or after line %d.\n", endmark, start_line);
            G__unexpectedEOF("G__fgetname():2");
            string.Set(i, 0);
            return(c);

         case '*':
         case '&':
            if (i > offset && ' ' == string[i-1] && nest && single_quote == 0 && double_quote == 0)
               --i;
            break;

         case ',':
            pp = string + i + 1;
            break;

         default:
            spaceflag = 1;
#ifdef G__MULTIBYTE
            if (G__IsDBCSLeadByte(c) && !ignoreflag) {
               string.Set(i++, c);
               c = G__fgetc() ;
               G__CheckDBCS2ndByte(c);
            }
#endif
            break;
      }

      if (ignoreflag == 0) {
         if (pflag && (isalpha(c) || '_' == c)) {
            string.Set(i++, ' ');
         }
         pflag = 0;
         string.Set(i++, c);
         G__CHECK(G__SECURE_BUFFER_SIZE, i >= G__LONGLINE, return(EOF));
      }

   }
   while (flag == 0) ;

   if (isspace(c)) {
      c = G__fgetspace();
      l = 0;
      flag = 0;
      while ((prev = endmark[l++]) != '\0') {
         if (c == prev) {
            flag = 1;
         }
      }
      if (!flag) {
         if ('<' == c) {
            if (strncmp(string + offset, "operator", 8) == 0) string.Set(i++, c);
            flag = ignoreflag = 0;
            goto backtoreadtemplate;
         }
         fseek(G__ifile.fp, -1, SEEK_CUR);
         if (G__dispsource) G__disp_mask = 1;
         c = ' ';
      }
   }

   string.Set(i, 0);

   return(c);
}

//______________________________________________________________________________
static char* G__get_previous_name(G__FastAllocString& string, size_t start, size_t offset) {
   ++start;
   while (start > offset) {
      char c = string[start - 1];
      if (c == ':' && start - 1 > offset && string[start - 1] == ':') {
         --start;
      } else if (!G__IsIdentifier(c)) {
         return string + start;
      }
      --start;
   }
   return string + start;
}

//______________________________________________________________________________
static int G__fgetstream_newtemplate_internal(G__FastAllocString& string,
                                                    size_t offset,
                                                    const char *endmark,
                                                    bool parseTemplate)
{
   // Work horse for G__fgetstream_newtemplate() and G__fgetstream_new().
   // See G__fgetstream_newtemplate() for emaning of parameters.
   size_t i = offset;
   size_t l = 0;
   int c;
   int nest = 0;
   bool single_quote = false;
   bool double_quote = false;
   bool breakflag = false;
   bool ignoreflag = false;
   bool commentflag = false;
   int start_line = G__ifile.line_number;

   do {
      ignoreflag = false;
      c = G__fgetc() ;

      if (nest <= 0 && !single_quote && !double_quote) {
         l = 0;
         int prev;
         while ((prev = endmark[l++]) != '\0') {
            if (c == prev) {
               breakflag = true;
               ignoreflag = true;
            }
         }
      }

      bool next_single_quote = single_quote;
      bool next_double_quote = double_quote;

      switch (c) {
         case '\n':
         case '\r':
            if (i && (single_quote == 0) && (double_quote == 0) && '\\' == string[i-1]) {
               // This is a line continuation, we just ignore it.
               --i;
               ignoreflag = true;
               breakflag = false; // Undo a possible marker match.
               break;
            }
         case ' ':
         case '\f':
         case '\t':
            commentflag = false;
            if (!single_quote && !double_quote) {
               c = ' ';
            }
            break;
         case ',':
            /* may be following line is needed. 1999/5/31 */
            /* if(i>2 && ' '==string[i-1] && isalnum(string[i-2])) --i; */
            break;

         case '<':
            if (parseTemplate && !single_quote && !double_quote) {
               string.Set(i, 0);
               char* prevIdentifier = i ? G__get_previous_name(string, i - 1, offset) : 0;
               if (prevIdentifier && prevIdentifier[0]
                   && G__defined_templateclass(prevIdentifier)){
                  ++nest;
               }
            }
            break;

         case '{':
         case '(':
         case '[':
            if (!single_quote && !double_quote) {
               ++nest;
            }
            break;
         case '>':
            if (parseTemplate && !single_quote && !double_quote) {
               if (0 == nest || (i && '-' == string[i - 1])) {
                  break;
               } else if (nest && i && '>' == string[i - 1]) {
                  string.Set(i++, ' ');
               }
               --nest;
            }
            break;
         case '}':
         case ')':
         case ']':
            if (!single_quote && !double_quote) {
               /* may be following line is needed. 1999/5/31 */
               /* if(i>2 && ' '==string[i-1] && isalnum(string[i-2])) --i; */
               --nest;
            }
            break;
         case '"':
            if (!single_quote) {
               next_double_quote = !double_quote;
            }
            break;
         case '\'':
            if (!double_quote) {
               next_single_quote = !single_quote;
            }
            break;

         case '\\':
            if (!ignoreflag) {
               string.Set(i++, c);
               c = G__fgetc() ;
               if ( (c=='\n' || c=='\r') && (single_quote == 0) && (double_quote == 0)) {
                  // This is a line continuation, we just ignore both the \ and \n.
                  --i;
                  ignoreflag = true;
               }
            }
            break;

         case '/':
            if (!double_quote && !single_quote && i > offset && string[i-1] == '/' &&
                  commentflag) {
               --i;
               G__fignoreline();
               ignoreflag = true;
            }
            else {
               commentflag = true;
            }
            break;

         case '*':
            /* comment */
            if (!double_quote && !single_quote) {
               if (i > offset && string[i-1] == '/' && commentflag) {
                  G__skip_comment();
                  --i;
                  ignoreflag = true;
               }
            }
            break;

         case '#':
            if (!single_quote && !double_quote && (i == offset || string[i-1] != '$')) {
               G__pp_command();
               ignoreflag = true;
#ifdef G__TEMPLATECLASS
               c = ' ';
#endif
            }
            break;

         case EOF:
            G__fprinterr(G__serr, "Error: Missing one of '%s' expected at or after line %d.\n", endmark, start_line);
            G__unexpectedEOF("G__fgetstream_newtemplate():2");
            string.Set(i, 0);
            return(c);

         case '=':
            break;

#ifdef G__MULTIBYTE
         default:
            if (G__IsDBCSLeadByte(c) && !ignoreflag) {
               string.Set(i++, c);
               c = G__fgetc() ;
               G__CheckDBCS2ndByte(c);
            }
            break;
#endif
      }

      if (!ignoreflag) {
         // i > 0, not i > offset: we care even about previous call's content of string
         if (i > 0 && !single_quote && !double_quote && string[i - 1] == ' ') {
            // We want to append c, but the trailing char is a space.
            if (c == ' ') --i; // replace ' ' by ' '
            else if (i == 1) {
               // string is " " - remove leading space.
               --i;
            } else {
               char prev = string[i - 2];
               // We only keep spaces between "identifiers" like "new const long long"
               // and between '> >'
               if ((G__IsIdentifier(prev) && G__IsIdentifier(c)) || (prev == '>' && c == '>')) {
               } else {
                  // replace previous ' '
                  --i;
               }
            }
         }
         string.Set(i++, c);
      }

      single_quote = next_single_quote;
      double_quote = next_double_quote;
   }
   while (!breakflag) ;

   if (i > 0 && string[i - 1] == ' ') --i;
   string.Set(i, 0);

   return(c);
}

//______________________________________________________________________________
int G__fgetstream_newtemplate(G__FastAllocString& string, size_t offset, const char *endmark)
{
   //  char *string       : string until the endmark appears
   //  char *endmark      : specify endmark characters
   // 
   //   read source file until specified endmark char appears, keeping a space after the 'new' keyword.
   // 
   //  1) read source file and store char to char *string.
   //    If char is space char which is not surrounded by quoatation
   //    it is not stored into char *string.
   //  2) When one of endmark char appears or parenthesis nesting of
   //    parenthesis gets negative , like '())' , stop reading and
   //    return the last char.
   // 
   //   *endmark=";"
   //      '  ab cd e f g;hijklm '
   //       -------------^          *string="abcdefg"; return(';');
   // 
   //   *endmark=";"
   //      ' abc );'
   //       -----^    *string="abc"; return(')');
   // 
   //      'abc=new xxx;'
   //      'func(new xxx);'
   // 

   return G__fgetstream_newtemplate_internal(string, offset, endmark, true);
}

//______________________________________________________________________________
int G__fgetstream_template(G__FastAllocString& string, size_t offset, const char *endmark)
{
   //  char *string       : string until the endmark appears
   //  char *endmark      : specify endmark characters
   // 
   //   read source file until specified endmark char appears.
   // 
   //  1) read source file and store char to char *string.
   //    If char is space char which is not surrounded by quoatation
   //    it is not stored into char *string.
   //  2) When one of endmark char appears or parenthesis nesting of
   //    parenthesis gets negative , like '())' , stop reading and
   //    return the last char.
   // 
   //   *endmark=";"
   //      '  ab cd e f g;hijklm '
   //       -------------^          *string="abcdefg"; return(';');
   // 
   //   *endmark=";"
   //      ' abc );'
   //       -----^    *string="abc"; return(')');
   // 
   size_t i = offset, l;
   int c, prev;
   short nest = 0, single_quote = 0, double_quote = 0, flag = 0, ignoreflag;
   int commentflag = 0;
   char *pp = string + offset;
   int pflag = 0;
   int start_line = G__ifile.line_number;

   do {
      ignoreflag = 0;
      c = G__fgetc() ;

      if ((nest == 0) && (single_quote == 0) && (double_quote == 0)) {
         l = 0;
         while ((prev = endmark[l++]) != '\0') {
            if (c == prev) {
               flag = 1;
               ignoreflag = 1;
            }
         }
      }

      switch (c) {
         case '\n':
         case '\r':
            if (i && (single_quote == 0) && (double_quote == 0) && '\\' == string[i-1]) {
               // This is a line continuation, we just ignore it.
               --i;
               ignoreflag = 1;
               flag = 0; // Undo a possible marker match.
               break;
            }
         case ' ':
         case '\f':
         case '\t':
            commentflag = 0;
            if ((single_quote == 0) && (double_quote == 0)) {
               string.Set(i, 0);
               if (G__isstoragekeyword(pp)) {
                  if (G__iscpp && strcmp("typename", pp) == 0) {
                     i -= 8;
                     c = ' ';
                     ignoreflag = 1;
                  }
                  else {
                     pp = string + i + 1;
                     c = ' ';
                  }
                  break;
               }
               else if (i && '*' == string[i-1]) {
                  pflag = 1;
               }
#define G__OLDIMPLEMENTATION1894
               ignoreflag = 1;
            }
            break;
         case '<':
#ifdef G__OLDIMPLEMENTATION1721_YET
            if ((single_quote == 0) && (double_quote == 0)) {
               string.Set(i, 0);
               if (G__defined_templateclass(pp)) ++nest;
               pp = string + i + 1;
            }
            break;
#endif
            // Fall through when G__OLDIMPLEMENTATION1721_YET is not defined.
         case '{':
         case '(':
         case '[':
            if ((single_quote == 0) && (double_quote == 0)) {
               pp = string + i + 1;
               nest++;
            }
            break;
         case '>':
            if (i && '-' == string[i-1]) break; /* need to test for >> ??? */
         case '}':
         case ')':
         case ']':
            if ((single_quote == 0) && (double_quote == 0)) {
               if (i > 2 && ' ' == string[i-1] && G__IsIdentifier(string[i-2])) --i;
               nest--;
               if (nest < 0) {
                  flag = 1;
                  ignoreflag = 1;
               }
               else if ('>' == c && i && '>' == string[i-1]) {
                  /* A<A<int> > */
                  string.Set(i++, ' ');
               }
            }
            break;
         case '"':
            if (single_quote == 0) double_quote ^= 1;
            break;
         case '\'':
            if (double_quote == 0) single_quote ^= 1;
            break;

         case '\\':
            if (ignoreflag == 0) {
               string.Set(i++, c);
               c = G__fgetc();
               if ( (c=='\n' || c=='\r') && (single_quote == 0) && (double_quote == 0)) {
                  // This is a line continuation, we just ignore both the \ and \n.
                  --i;
                  ignoreflag = 1;
               }
            }
            break;

         case '/':
            if (0 == double_quote && 0 == single_quote && i > offset && string[i-1] == '/' &&
                  commentflag) {
               G__fignoreline();
               --i;
               ignoreflag = 1;
            }
            else {
               commentflag = 1;
            }
            break;

         case '&':
            if (i > offset && ' ' == string[i-1] && nest && single_quote == 0 && double_quote == 0)
               --i;
            break;

         case '*':
            /* comment */
#ifndef G__OLDIMPLEMENTATION1864
            if (0 == double_quote && 0 == single_quote && i > offset) {
               if (string[i-1] == '/' && commentflag) {
                  G__skip_comment();
                  --i;
                  ignoreflag = 1;
               }
               else
                  if (i > 2 &&
                      isspace(string[i-1]) &&
                      G__IsIdentifier(string[i-2])
                     ) {
                     --i;
                  }
            }

#else
            if (0 == double_quote && 0 == single_quote && i > offset && string[i-1] == '/' &&
                  commentflag) {
               G__skip_comment();
               --i;
               ignoreflag = 1;
            }
#endif
            break;

         case '#':
            if (single_quote == 0 && double_quote == 0 && (i == offset || string[i-1] != '$')) {
               G__pp_command();
               ignoreflag = 1;
#ifdef G__TEMPLATECLASS
               c = ' ';
#endif
            }
            break;

         case EOF:
            G__fprinterr(G__serr, "Error: Missing one of '%s' expected at or after line %d.\n", endmark, start_line);
            G__unexpectedEOF("G__fgetstream_template():2");
            string.Set(i, 0);
            return(c);
            /* break; */


         case ',':
            if (i > 2 && ' ' == string[i-1] && G__IsIdentifier(string[i-2])) --i;
            pp = string + i + 1;
            break;

#ifdef G__MULTIBYTE
         default:
            if (G__IsDBCSLeadByte(c) && !ignoreflag) {
               string.Set(i++, c);
               c = G__fgetc() ;
               G__CheckDBCS2ndByte(c);
            }
            break;
#endif
      }

      if (ignoreflag == 0) {
         if (pflag && (isalpha(c) || '_' == c)) {
            string.Set(i++, ' ');
         }
         pflag = 0;
         string.Set(i++, c);
         G__CHECK(G__SECURE_BUFFER_SIZE, i >= G__LONGLINE, return(EOF));
      }

   }
   while (flag == 0) ;

   string.Set(i, 0);

   return(c);
}

//______________________________________________________________________________
int G__getstream_template(const char* source, int* isrc,G__FastAllocString&  string, size_t offset, const char* endmark)
{
   //  char *source;      : source string. If NULL, read from input file
   //  int *isrc;         : char position of the *source if source!=NULL
   //  char *string       : substring until the endmark appears
   //  char *endmark      : specify endmark characters
   // 
   // 
   //   Get substring of char *source; until one of endmark char is found.
   //  Return string is not used.
   //   Only used in G__getexpr() to handle 'cond?true:faulse' opeartor.
   // 
   //    char *endmark=";";
   //    char *source="abcdefg * hijklmn ;   "
   //                  ------------------^      *string="abcdefg*hijklmn"
   // 
   //    char *source="abcdefg * hijklmn) ;   "
   //                  -----------------^       *string="abcdefg*hijklmn"
   // 
   size_t i = offset, l;
   int c, prev;
   short nest = 0, single_quote = 0, double_quote = 0, flag = 0, ignoreflag;
   char *pp = string + offset;
   int pflag = 0;
   int start_line = G__ifile.line_number;

   do {
      ignoreflag = 0;
      c = source[(*isrc)++] ;

      if ((nest == 0) && (single_quote == 0) && (double_quote == 0)) {
         l = 0;
         while ((prev = endmark[l++]) != '\0') {
            if (c == prev) {
               flag = 1;
               ignoreflag = 1;
            }
         }
      }

      switch (c) {
         case '"':
            if (single_quote == 0) double_quote ^= 1;
            break;
         case '\'':
            if (double_quote == 0) single_quote ^= 1;
            break;
         case '<':
         case '{':
         case '(':
         case '[':
            if ((single_quote == 0) && (double_quote == 0)) {
               pp = string + i + 1;
               nest++;
            }
            break;
         case '>':
            if (i && '-' == string[i-1]) break;
         case '}':
         case ')':
         case ']':
            if ((single_quote == 0) && (double_quote == 0)) {
               if (i > 2 && ' ' == string[i-1] && G__IsIdentifier(string[i-2])) --i;
               nest--;
               if (nest < 0) {
                  flag = 1;
                  ignoreflag = 1;
               }
               else if ('>' == c && i && '>' == string[i-1]) {
                  /* A<A<int> > */
                  string.Set(i++, ' ');
               }
            }
            break;
         case '\n':
         case '\r':
            if (i && (single_quote == 0) && (double_quote == 0) && '\\' == string[i-1]) {
               // This is a line continuation, we just ignore it.
               --i;
               ignoreflag = 1;
               flag = 0; // Undo a possible marker match.
               break;
            }
         case ' ':
         case '\f':
         case '\t':
            if ((single_quote == 0) && (double_quote == 0)) {
               string.Set(i, 0);
               if (G__isstoragekeyword(pp)) {
                  if (G__iscpp && strcmp("typename", pp) == 0) {
                     i -= 8;
                     c = ' ';
                     ignoreflag = 1;
                  }
                  else {
                     pp = string + i + 1;
                     c = ' ';
                  }
                  break;
               }
               else if (i>offset && '*' == string[i-1]) {
                  pflag = 1;
               }
               ignoreflag = 1;
            }
            break;
         case '\0':
            /* if((single_quote==0)&&(double_quote==0)) { */
            flag = 1;
            ignoreflag = 1;
            /* } */
            break;
         case EOF:
            G__fprinterr(G__serr, "Error: Missing one of '%s' expected at or after line %d.\n", endmark, start_line);
            G__unexpectedEOF("G__getstream()");
            string.Set(i, 0);
            break;


         case ',':
            if (i > 2 && ' ' == string[i-1] && G__IsIdentifier(string[i-2])) --i;
            pp = string + i + 1;
            break;

#ifdef G__MULTIBYTE
         default:
            if (G__IsDBCSLeadByte(c) && !ignoreflag) {
               string.Set(i++, c);
               c = G__fgetc() ;
               G__CheckDBCS2ndByte(c);
            }
            break;
#endif
      }

      if (ignoreflag == 0) {
         if (pflag && (isalpha(c) || '_' == c)) {
            string.Set(i++, ' ');
         }
         pflag = 0;
         string.Set(i++, c);
      }

   }
   while (flag == 0) ;

   string.Set(i, 0);

   return(c);
}

//______________________________________________________________________________
int G__fgetspace()
{
   // -- Read source file until a non-space character appears, may return EOF character.
   // 
   //  1) read until non-space char appears
   //  2) return first non-space char
   // 
   //      '         abcd...'
   //       ---------^     return('a');
   // 
   int c = 0;
   short flag = 0;
   int start_line = G__ifile.line_number;
   do {
      c = G__fgetc();
      switch (c) {
         case ' ':
         case '\t':
         case '\n':
         case '\r':
         case '\f':
            // -- Whitespace, continue scanning.
            break;
         case '/':
            // -- Possibly a comment, if so, handle and continue scanning, otherwise stop.
            // Look ahead at the next character.
            c = G__fgetc();
            switch (c) {
               case '*':
                  // -- C style comment.
                  G__skip_comment();
                  break;
               case '/':
                  // -- C++ style comment.
                  G__fignoreline();
                  break;
               default:
                  // -- Not a comment, undo the lookahead.
                  // Backup file position by one character.
                  fseek(G__ifile.fp, -1, SEEK_CUR);
                  // Undo any line number increment.
                  if (c == '\n') {
                     --G__ifile.line_number;
                  }
                  // Do not print character a second time.
                  if (G__dispsource) {
                     G__disp_mask = 1;
                  }
                  // We saw a slash.
                  c = '/';
                  // We saw a non-whitespace character, flag all done.
                  flag = 1;
                  break;
            }
            break;
         case '#':
            // -- Prepreocessor command, handle and continue scanning.
            G__pp_command();
#ifdef G__TEMPLATECLASS
            c = ' ';
#endif // G__TEMPLATECLASS
            break;
         case EOF:
            // -- Oops, end of file, stop and return EOF.
            G__fprinterr(G__serr, "Error: Missing whitespace at or after line %d.\n", start_line);
            G__unexpectedEOF("G__fgetspace():2");
            return c;
         default:
            // -- Not whitespace, we are done.
            flag = 1;
            break;
      }
   }
   while (!flag);
   return c;
}

//______________________________________________________________________________
int G__fgetspace_peek()
{
   // -- Read source file until a non-space character appears, may return EOF character.
   // FIXME: We do not handle macro expansion!
   //
   // First, remember the current file position.
   fpos_t store_fpos;
   fgetpos(G__ifile.fp, &store_fpos);
   // Now scan.
   int c = 0;
   int flag = 0;
   do {
      c = fgetc(G__ifile.fp);
      switch (c) {
         case ' ':
         case '\t':
         case '\n':
         case '\r':
         case '\f':
            // -- Whitespace, continue scanning.
            break;
         case '/':
            // -- Possibly a comment, if so, handle and continue scanning, otherwise stop.
            // Look ahead at the next character.
            c = fgetc(G__ifile.fp);
            switch (c) {
               case '*':
                  // -- C style comment.
                  G__skip_comment_peek();
                  break;
               case '/':
                  // -- C++ style comment.
                  G__fignoreline_peek();
                  break;
               default:
                  // -- Not a comment, undo the lookahead.
                  // Backup file position by one character.
                  fseek(G__ifile.fp, -1, SEEK_CUR);
                  // We saw a slash.
                  c = '/';
                  // We saw a non-whitespace character, flag all done.
                  flag = 1;
                  break;
            }
            break;
         default:
            // -- Not whitespace, we are done.
            flag = 1;
            break;
      }
   }
   while (!flag);
   // All done, restore previous input file position.
   fsetpos(G__ifile.fp, &store_fpos);
   return c;
}

//______________________________________________________________________________
int G__fgetvarname(G__FastAllocString& string, size_t offset, const char *endmark)
{
   size_t i = offset;
   size_t l = 0;
   int c;
   int nest = 0;
   bool single_quote = false;
   bool double_quote = false;
   bool breakflag = false;
   bool ignoreflag = false;
   bool commentflag = false;
   bool haveid = false;
#ifdef G__TEMPLATEMEMFUNC
   int tmpltlevel = 0;
   bool operGt = false;
#endif
   int start_line = G__ifile.line_number;

   do {
      ignoreflag = 0;
      c = G__fgetc() ;

      if ((nest <= 0) && !tmpltlevel && (!single_quote) && (!double_quote)) {
         l = 0;
         int prev;
         while ((prev = endmark[l++]) != '\0') {
            if (c == prev) {
               breakflag = true;
               ignoreflag = true;
            }
         }
      }

      bool next_single_quote = single_quote;
      bool next_double_quote = double_quote;

      switch (c) {
         case ' ':
         case '\t':
         case '\n':
         case '\r':
         case '\f':
            commentflag = false;
            if (!single_quote && !double_quote) {
               c = ' ';
               if (!nest && !tmpltlevel && haveid) {
                  breakflag = true;
               }
            }
            break;
#ifdef G__TEMPLATEMEMFUNC
         case '<':
            if (!single_quote && !double_quote) {
               if (operGt || (8 == i - offset && strncmp("operator", string() + offset, 8) == 0)
                   || (9 == i - offset && (strncmp("&operator", string() + offset, 9) == 0 ||
                                           strncmp("*operator", string() + offset, 9) == 0))
                   ) {
                  operGt = true;
                  break;
               } else {
                  string.Set(i, 0);
                  char* prevIdentifier = i ? G__get_previous_name(string, i - 1, offset) : 0;
                  if (prevIdentifier && prevIdentifier[0]){
                     ++tmpltlevel;
                  }
               }
            }
            break;
#endif
         case '{':
         case '(':
         case '[':
            if (!single_quote && !double_quote) {
               ++nest;
            }
            break;
#ifdef G__TEMPLATEMEMFUNC
         case '>':
            if (!single_quote && !double_quote) {
               if (!tmpltlevel) {
                  break;
               } else if (nest && i && '>' == string[i - 1]) {
                  string.Set(i++, ' ');
               }
               --tmpltlevel;
               if (tmpltlevel < 0) {
                  breakflag = 1;
                  ignoreflag = 1;
               }
            }
            break;
#endif
         case '}':
         case ')':
         case ']':
            if (!single_quote && !double_quote) {
               nest--;
               if (nest < 0) {
                  breakflag = 1;
                  ignoreflag = 1;
               }
            }
            break;
         case '"':
            if (!single_quote) {
               next_double_quote = !double_quote;
            }
            break;
         case '\'':
            if (!double_quote) {
               next_single_quote = !single_quote;
            }
            break;
         case '/':
            if (!single_quote && !double_quote && i > offset && string[i-1] == '/' &&
                  commentflag) {
               --i;
               G__fignoreline();
               ignoreflag = true;
            } else {
               commentflag = true;
            }
            break;

         case '#':
            if (!single_quote && !double_quote && (i == offset || string[i-1] != '$')) {
               G__pp_command();
               ignoreflag = true;
#ifdef G__TEMPLATECLASS
               c = ' ';
#endif
            }
            break;

         case EOF:
            G__fprinterr(G__serr, "Error: Missing one of '%s' expected at or after line %d.\n", endmark, start_line);
            G__unexpectedEOF("G__fgetvarname():2");
            string.Set(i, 0);
            return(c);

         case '*':
             /* comment */
            if (!double_quote && !single_quote) {
               if (commentflag && i > offset && string[i-1] == '/') {
                  G__skip_comment();
                  --i;
                  ignoreflag = true;
                  break;
               }
            }
            // intentional fall-through to default

         case ',':
            /* fall through... */

         default:
            haveid = true;
#ifdef G__MULTIBYTE
            if (G__IsDBCSLeadByte(c) && !ignoreflag) {
               string.Set(i++, c);
               c = G__fgetc() ;
               G__CheckDBCS2ndByte(c);
            }
#endif
            break;
      }

      if (!ignoreflag) {
         // i > 0, not i > offset: we care even about previous call's content of string
         if (i > 0 && !single_quote && !double_quote && string[i - 1] == ' ') {
            // We want to append c, but the trailing char is a space.
            if (c == ' ') --i; // replace ' ' by ' '
            else if (i == 1) {
               // string is " " - remove leading space.
               --i;
            } else {
               char prev = string[i - 2];
               // We only keep spaces between "identifiers" like "new const long long"
               // and between '> >'
               if ((G__IsIdentifier(prev) && G__IsIdentifier(c)) || (prev == '>' && c == '>')) {
               } else {
                  // replace previous ' '
                  --i;
               }
            }
         }
         string.Set(i++, c);
      }

      single_quote = next_single_quote;
      double_quote = next_double_quote;
   }
   while (!breakflag) ;

   if (i > 0 && string[i - 1] == ' ') --i;
   string.Set(i, 0);

   return(c);
}

//______________________________________________________________________________
int G__fgetname(G__FastAllocString& string, size_t offset, const char *endmark)
{
   //  char *string       : string until the endmark appears
   //  char *endmark      : specify endmark characters
   // 
   //    read one non-space char string upto next space char or endmark
   //   char.
   // 
   //  1) skip space char until non space char appears
   //  2) Store non-space char to char *string. If space char is surrounded by
   //    quotation, it is stored.
   //  3) if space char or one of endmark char which is not surrounded by
   //    quotation appears, stop reading and return the last char.
   // 
   // 
   //    '     azAZ09*&^%/     '
   //     ----------------^        return(' ');
   // 
   //  if ";" is given as end mark
   //    '     azAZ09*&^%/;  '
   //     ----------------^        return(';');
   // 
   size_t i = offset, l;
   int c, prev;
   short single_quote = 0, double_quote = 0, flag = 0, spaceflag, ignoreflag;
   int start_line = G__ifile.line_number;

   spaceflag = 0;

   do {
      ignoreflag = 0;
      c = G__fgetc() ;

      if ((single_quote == 0) && (double_quote == 0)) {
         l = 0;
         while ((prev = endmark[l++]) != '\0') {
            if (c == prev) {
               flag = 1;
               ignoreflag = 1;
            }
         }
      }

      switch (c) {
         case ' ':
         case '\t':
         case '\n':
         case '\r':
         case '\f':
            if ((single_quote == 0) && (double_quote == 0)) {
               ignoreflag = 1;
               if (spaceflag != 0) flag = 1;
            }
            break;
         case '"':
            if (single_quote == 0) {
               spaceflag = 1;
               double_quote ^= 1;
            }
            break;
         case '\'':
            if (double_quote == 0) {
               spaceflag = 1;
               single_quote ^= 1;
            }
            break;
            /*
              case '\0':
              flag=1;
              ignoreflag=1;
              break;
              */
         case '/':
            if ((single_quote == 0) && (double_quote == 0)) {
               /* comment */
               string.Set(i++, c);

               c = G__fgetc();
               switch (c) {
                  case '*':
                     G__skip_comment();
                     --i;
                     ignoreflag = 1;
                     break;
                  case '/':
                     G__fignoreline();
                     --i;
                     ignoreflag = 1;
                     break;
                  default:
                     fseek(G__ifile.fp, -1, SEEK_CUR);
                     if (G__dispsource) G__disp_mask = 1;
                     spaceflag = 1;
                     ignoreflag = 1;
                     break;
               }
            }

            break;

         case '#':
            if (single_quote == 0 && double_quote == 0 && (i == offset || string[i-1] != '$')) {
               G__pp_command();
               ignoreflag = 1;
#ifdef G__TEMPLATECLASS
               c = ' ';
#endif
            }
            break;

         case EOF:
            G__fprinterr(G__serr, "Error: Missing one of '%s' expected at or after line %d.\n", endmark, start_line);
            G__unexpectedEOF("G__fgetname():2");
            string.Set(i, 0);
            return(c);
         default:
            spaceflag = 1;
#ifdef G__MULTIBYTE
            if (G__IsDBCSLeadByte(c) && !ignoreflag) {
               string.Set(i++, c);
               c = G__fgetc() ;
               G__CheckDBCS2ndByte(c);
            }
#endif
            break;
      }

      if (ignoreflag == 0) {
         string.Set(i++, c);
      }

   }
   while (flag == 0) ;

   string.Set(i, 0);

   return(c);
}

//______________________________________________________________________________
int G__getname(const char* source, int* isrc, char* string, const char* endmark)
{
   //  char *string       : string until the endmark appears
   //  char *endmark      : specify endmark characters
   // 
   //    read one non-space char string upto next space char or endmark
   //   char.
   // 
   //  1) skip space char until non space char appears
   //  2) Store non-space char to char *string. If space char is surrounded by
   //    quotation, it is stored.
   //  3) if space char or one of endmark char which is not surrounded by
   //    quotation appears, stop reading and return the last char.
   // 
   // 
   //    '     azAZ09*&^%/     '
   //     ----------------^        return(' ');
   // 
   //  if ";" is given as end mark
   //    '     azAZ09*&^%/;  '
   //     ----------------^        return(';');
   // 
   short i = 0, l;
   int c, prev;
   short single_quote = 0, double_quote = 0, flag = 0, spaceflag, ignoreflag;
   int start_line = G__ifile.line_number;

   spaceflag = 0;

   do {
      ignoreflag = 0;
      c = source[(*isrc)++] ;

      if ((single_quote == 0) && (double_quote == 0)) {
         l = 0;
         while ((prev = endmark[l++]) != '\0') {
            if (c == prev) {
               flag = 1;
               ignoreflag = 1;
            }
         }
      }

      switch (c) {
         case ' ':
         case '\t':
         case '\n':
         case '\r':
         case '\f':
            if ((single_quote == 0) && (double_quote == 0)) {
               ignoreflag = 1;
               if (spaceflag != 0) flag = 1;
            }
            break;
         case '"':
            if (single_quote == 0) {
               spaceflag = 1;
               double_quote ^= 1;
            }
            break;
         case '\'':
            if (double_quote == 0) {
               spaceflag = 1;
               single_quote ^= 1;
            }
            break;
            /*
              case '\0':
              flag=1;
              ignoreflag=1;
              break;
              */
#ifdef G__NEVER
         case '/':
            if ((single_quote == 0) && (double_quote == 0)) {
               /* comment */
               string[i++] = c ;

               c = source[(*isrc)++] ;
               switch (c) {
                  case '*':
                     G__skip_comment();
                     --i;
                     ignoreflag = 1;
                     break;
                  case '/':
                     G__fignoreline();
                     --i;
                     ignoreflag = 1;
                     break;
                  default:
                     fseek(G__ifile.fp, -1, SEEK_CUR);
                     if (G__dispsource) G__disp_mask = 1;
                     spaceflag = 1;
                     ignoreflag = 1;
                     break;
               }
            }

            break;

         case '#':
            if (single_quote == 0 && double_quote == 0 && (i == offset || string[i-1] != '$')) {
               G__pp_command();
               ignoreflag = 1;
#ifdef G__TEMPLATECLASS
               c = ' ';
#endif
            }
            break;
#endif

         case EOF:
            G__fprinterr(G__serr, "Error: Missing one of '%s' expected at or after line %d.\n", endmark, start_line);
            G__unexpectedEOF("G__fgetname():2");
            string[i] = '\0';
            return(c);
         default:
            spaceflag = 1;
#ifdef G__MULTIBYTE
            if (G__IsDBCSLeadByte(c) && !ignoreflag) {
               string[i++] = c;
               c = source[(*isrc)++] ;
               G__CheckDBCS2ndByte(c);
            }
#endif
            break;
      }

      if (ignoreflag == 0) {
         string[i++] = c ;
         G__CHECK(G__SECURE_BUFFER_SIZE, i >= G__LONGLINE, return(EOF));
      }

   }
   while (flag == 0) ;

   string[i] = '\0';

   return(c);
}

//______________________________________________________________________________
static size_t G__getfullpath(G__FastAllocString &string, char* pbegin, size_t i)
{
   int tagnum = -1, typenum;
   string.Set(i, '\0');
   if (0 == pbegin[0]) return(i);
   typenum = G__defined_typename(pbegin);
   if (-1 == typenum) tagnum = G__defined_tagname(pbegin, 1);
   if ((-1 != typenum && -1 != G__newtype.parent_tagnum[typenum]) ||
         (-1 != tagnum  && -1 != G__struct.parent_tagnum[tagnum])) {
      if ( (size_t)(pbegin - string) < string.Capacity() ) // sanity check
      {
         string.Replace(pbegin - string, G__type2string(0, tagnum, typenum, 0, 0));  
      }
      i = strlen(string);
   }
   return(i);
}

//______________________________________________________________________________
int G__fdumpstream(G__FastAllocString& string, size_t offset, const char *endmark)
{
   //  char *string       : string until the endmark appears
   //  char *endmark      : specify endmark characters
   // 
   //   This function is used only for reading pointer to function arguments.
   //     type (*)(....)  type(*p2f)(....)
   //
   size_t i = offset, l;
   int c, prev;
   short nest = 0, single_quote = 0, double_quote = 0, flag = 0, ignoreflag;
   int commentflag = 0;
   char *pbegin = string + offset;
   int tmpltnest = 0;
   int start_line = G__ifile.line_number;

   do {
      ignoreflag = 0;
      c = G__fgetc() ;

      if ((nest == 0) && (single_quote == 0) && (double_quote == 0)) {
         l = 0;
         while ((prev = endmark[l++]) != '\0') {
            if (c == prev) {
               flag = 1;
               ignoreflag = 1;
            }
         }
      }

      switch (c) {
         case ' ':
         case '\t':
         case '\n':
         case '\r':
         case '\f':
            commentflag = 0;
            if ((single_quote == 0) && (double_quote == 0)) {
               c = ' ';
               if (i > offset && isspace(string[i-1])) {
                  ignoreflag = 1;
               }
               else {
                  i = G__getfullpath(string, pbegin, i);
               }
               if (tmpltnest == 0) pbegin = string + i + 1 - ignoreflag;
            }
            break;

         case '<':
            if ((single_quote == 0) && (double_quote == 0)) {
               string.Set(i, 0);
               if (G__defined_templateclass(pbegin)) ++tmpltnest;
            }
            break;
         case '>':
            if ((single_quote == 0) && (double_quote == 0)) {
               if (tmpltnest) --tmpltnest;
            }
            break;

         case '{':
         case '(':
         case '[':
            if ((single_quote == 0) && (double_quote == 0)) {
               nest++;
               pbegin = string + i + 1;
            }
            break;
         case '}':
         case ')':
         case ']':
            if ((single_quote == 0) && (double_quote == 0)) {
               nest--;
               if (nest < 0) {
                  flag = 1;
                  ignoreflag = 1;
               }
               i = G__getfullpath(string, pbegin, i);
               pbegin = string + i + 1 - ignoreflag;
            }
            break;
         case '"':
            if (single_quote == 0) {
               double_quote ^= 1;
            }
            break;
         case '\'':
            if (double_quote == 0) {
               single_quote ^= 1;
            }
            break;

         case '\\':
            if (ignoreflag == 0) {
               string.Set(i++, c);
               c = G__fgetc() ;
               if ( (c=='\n' || c=='\r') && (single_quote == 0) && (double_quote == 0)) {
                  // This is a line continuation, we just ignore both the \ and \n.
                  --i;
                  ignoreflag = 1;
               }
            }
            break;

            /*
              case '\0':
              flag=1;
              ignoreflag=1;
              break;
              */


         case '/':
            if (0 == double_quote && 0 == single_quote && i > offset && string[i-1] == '/' &&
                  commentflag) {
               G__fignoreline();
               --i;
               ignoreflag = 1;
            }
            else {
               commentflag = 1;
            }
            break;

         case '*':
            /* comment */
            if (0 == double_quote && 0 == single_quote && i > offset && string[i-1] == '/' &&
                  commentflag) {
               G__skip_comment();
               --i;
               ignoreflag = 1;
            }
            if (ignoreflag == 0) i = G__getfullpath(string, pbegin, i);
            pbegin = string + i + 1 - ignoreflag;
            break;

         case '&':
         case ',':
            i = G__getfullpath(string, pbegin, i);
            pbegin = string + i + 1;
            break;

         case '#':
            if (single_quote == 0 && double_quote == 0 && (i == offset || string[i-1] != '$')) {
               G__pp_command();
               ignoreflag = 1;
#ifdef G__TEMPLATECLASS
               c = ' ';
#endif
            }
            break;

         case EOF:
            G__fprinterr(G__serr, "Error: Missing one of '%s' expected at or after line %d.\n", endmark, start_line);
            G__unexpectedEOF("G__fdumpstream():2");
            string.Set(i, 0);
            return(c);

#ifdef G__MULTIBYTE
         default:
            if (G__IsDBCSLeadByte(c) && !ignoreflag) {
               string.Set(i++, c);
               c = G__fgetc() ;
               G__CheckDBCS2ndByte(c);
            }
            break;
#endif
      }

      if (ignoreflag == 0) {
         string.Set(i++, c);
      }

   }
   while (flag == 0) ;

   string.Set(i, 0);

   return(c);
}

//______________________________________________________________________________
int G__fgetstream(G__FastAllocString& string, size_t offset, const char *endmark)
{
   // -- Read source file until specified endmark char appears.
   //
   //  string: string until the endmark appears
   //  endmark: specify endmark characters
   // 
   //  1) read source file and store char to char *string.
   //    If char is space char which is not surrounded by quoatation
   //    it is not stored into char *string.
   //  2) When one of endmark char appears or parenthesis nesting of
   //    parenthesis gets negative , like '())' , stop reading and
   //    return the last char.
   // 
   //   *endmark=";"
   //      '  ab cd e f g;hijklm '
   //       -------------^          *string="abcdefg"; return(';');
   // 
   //   *endmark=";"
   //      ' abc );'
   //       -----^    *string="abc"; return(')');
   // 
   size_t i = offset;
   short l = 0;
   int c = 0;
   int prev = 0;
   short nest = 0;
   short single_quote = 0;
   short double_quote = 0;
   short flag = 0;
   short ignoreflag = 0;
   int commentflag = 0;
   int start_line = G__ifile.line_number;
   do {
      ignoreflag = 0;
      c = G__fgetc() ;
      if (!nest && !single_quote && !double_quote) {
         l = 0;
         while ((prev = endmark[l++]) != '\0') {
            if (c == prev) {
               flag = 1;
               ignoreflag = 1;
               break;
            }
         }
      }
      switch (c) {
         case '\f':
         case '\n':
         case '\r':
         case '\t':
         case ' ':
            commentflag = 0;
            if (!single_quote && !double_quote) {
               ignoreflag = 1;
            }
            break;
         case '{':
         case '(':
         case '[':
            if (!single_quote && !double_quote) {
               nest++;
            }
            break;
         case '}':
         case ')':
         case ']':
            if (!single_quote && !double_quote) {
               nest--;
               if (nest < 0) {
                  flag = 1;
                  ignoreflag = 1;
               }
            }
            break;
         case '"':
            if (!single_quote) {
               double_quote ^= 1;
            }
            break;
         case '\'':
            if (!double_quote) {
               single_quote ^= 1;
            }
            break;
         case '\\':
            if (!ignoreflag) {
               string.Set(i++, c);
               c = G__fgetc();
               if ( (c=='\n' || c=='\r') && (single_quote == 0) && (double_quote == 0)) {
                  // This is a line continuation, we just ignore both the \ and \n.
                  --i;
                  ignoreflag = 1;
               }
            }
            break;
         case '/':
            if (!double_quote && !single_quote && (i > offset) && (string[i-1] == '/') && commentflag) {
               G__fignoreline();
               --i;
               ignoreflag = 1;
               if (strchr(endmark, '\n')) {
                  c = '\n';
                  flag = 1;
               }
            }
            else {
               commentflag = 1;
            }
            break;
         case '*':
            // comment
            if (!double_quote && !single_quote && (i > offset) && (string[i-1] == '/') && commentflag) {
               G__skip_comment();
               --i;
               ignoreflag = 1;
            }
            break;
         case '#':
            if (!single_quote && !double_quote && !flag && (!i || string[i-1] != '$')) {
               G__pp_command();
               ignoreflag = 1;
#ifdef G__TEMPLATECLASS
               c = ' ';
#endif // G__TEMPLATECLASS
            }
            break;
         case EOF:
            G__fprinterr(G__serr, "Error: Missing one of '%s' expected at or after line %d.\n", endmark, start_line);
            G__unexpectedEOF("G__fgetstream():2");
            string.Set(i, 0);
            return c;
         // --
#ifdef G__MULTIBYTE
         default:
            if (G__IsDBCSLeadByte(c) && !ignoreflag) {
               string.Set(i++, c);
               c = G__fgetc() ;
               G__CheckDBCS2ndByte(c);
            }
            break;
#endif // G__MULTIBYTE
      }
      if (!ignoreflag) {
         string.Set(i++, c);
      }
   }
   while (!flag);
   string.Set(i, 0);
   return c;
}

//______________________________________________________________________________
void G__fgetstream_peek(char* string, int nchars)
{
   // -- Peak ahead upto nchars into source file.
   //
   //  string: result
   //  nchars: max number of characters to lookahead
   // 
   int i = 0;
   // First, remember the current file position.
   fpos_t store_fpos;
   fgetpos(G__ifile.fp, &store_fpos);
   for (; i < nchars; ++i) {
      int c = fgetc(G__ifile.fp);
      switch (c) {
         case EOF:
            string[i] = '\0';
            // All done, restore previous input file position.
            fsetpos(G__ifile.fp, &store_fpos);
            return;
         // --
#ifdef G__MULTIBYTE
         default:
            if (G__IsDBCSLeadByte(c)) {
               string[i++] = c;
               c = fgetc(G__ifile.fp) ;
               G__CheckDBCS2ndByte(c);
            }
            break;
#endif // G__MULTIBYTE
      }
      string[i] = c;
   }
   string[i] = '\0';
   // All done, restore previous input file position.
   fsetpos(G__ifile.fp, &store_fpos);
   return;
}

//______________________________________________________________________________
int G__fgetstream_new(G__FastAllocString& string, size_t offset, const char *endmark)
{
   // -- Read source file until specified endmark char appears, keep space after 'new' and 'const' keywords.
   //
   //  string: string until the endmark appears
   //  endmark: specify endmark characters
   // 
   //  1) read source file and store char to char *string.
   //    If char is space char which is not surrounded by quoatation
   //    it is not stored into char *string.
   //  2) When one of endmark char appears or parenthesis nesting of
   //    parenthesis gets negative , like '())' , stop reading and
   //    return the last char.
   // 
   //   *endmark=";"
   //      '  ab cd e f g;hijklm '
   //       -------------^          *string="abcdefg"; return(';');
   // 
   //   *endmark=";"
   //      ' abc );'
   //       -----^    *string="abc"; return(')');
   // 
   //      'abc=new xxx;'
   //      'func(new xxx);'
   //      'func(const int xxx);'
   // 

   return G__fgetstream_newtemplate_internal(string, offset, endmark, false);
}

//______________________________________________________________________________
int G__fgetstream_spaces(G__FastAllocString& string, size_t offset, const char *endmark)
{
   // -- Read source file until specified endmark char appears, retain whitespace (trimmed and collapsed).
   //
   //   string: string until the endmark appears
   //   endmark: specify endmark characters
   // 
   //   Just like G__fgetstream(), except that spaces are not
   //   completely removed.  Multiple spaces are, however, collapsed
   //   into a single space; leading and trailing spaces are also removed.
   // 
   //   *endmark=";"
   //      '  ab cd e f g;hijklm '
   //       -------------^          *string="ab cd e f g"; return(';');
   // 
   //   *endmark=";"
   //      ' abc );'
   //       -----^    *string="abc"; return(')');
   // 
   size_t i = offset;
   short l = 0;
   int c = 0;
   int prev = 0;
   int nest = 0;
   int single_quote = 0;
   int double_quote = 0;
   int flag = 0;
   int ignoreflag = 0;
   int commentflag = 0;
   int last_was_space = 0;
   int start_line = G__ifile.line_number;
   do {
      ignoreflag = 0;
      c = G__fgetc() ;
      if ((nest <= 0) && !single_quote && !double_quote) {
         l = 0;
         while ((prev = endmark[l++]) != '\0') {
            if (c == prev) {
               flag = 1;
               ignoreflag = 1;
            }
         }
      }
      switch (c) {
         case '\n':
         case '\r':
            if (i && (single_quote == 0) && (double_quote == 0) && '\\' == string[i-1]) {
               // This is a line continuation, we just ignore it.
               --i;
               ignoreflag = 1;
               flag = 0; // Undo a possible marker match.
               break;
            }
         case '\f':
         case '\t':
         case ' ':
            commentflag = 0;
            if (!single_quote && !double_quote) {
               c = ' ';
               if (last_was_space || !i)
                  ignoreflag = 1;
            }
            break;
         case '{':
         case '(':
         case '[':
            if (!single_quote && !double_quote) {
               ++nest;
            }
            break;
         case '}':
         case ')':
         case ']':
            if (!single_quote && !double_quote) {
               --nest;
            }
            break;
         case '"':
            if (!single_quote) {
               double_quote ^= 1;
            }
            break;
         case '\'':
            if (!double_quote) {
               single_quote ^= 1;
            }
            break;
         case '\\':
            if (!ignoreflag) {
               string.Set(i++, c);
               c = G__fgetc();
               if ( (c=='\n' || c=='\r') && (single_quote == 0) && (double_quote == 0)) {
                  // This is a line continuation, we just ignore both the \ and \n.
                  --i;
                  ignoreflag = 1;
               }
            }
            break;
         case '/':
            if (!double_quote && !single_quote && (i > offset) && (string[i-1] == '/') && commentflag) {
               --i;
               G__fignoreline();
               ignoreflag = 1;
            }
            else {
               commentflag = 1;
            }
            break;
         case '*':
            // comment
            if (!double_quote && !single_quote && (i > offset) && (string[i-1] == '/') && commentflag) {
               G__skip_comment();
               --i;
               ignoreflag = 1;
            }
            break;
         case '#':
            if (!single_quote && !double_quote && (!i || (string[i-1] != '$'))) {
               G__pp_command();
               ignoreflag = 1;
#ifdef G__TEMPLATECLASS
               c = ' ';
#endif
               // --
            }
            break;
         case EOF:
            G__fprinterr(G__serr, "Error: Missing one of '%s' expected at or after line %d.\n", endmark, start_line);
            G__unexpectedEOF("G__fgetstream_new():2");
            string.Set(i, 0);
            return c;
         // --
#ifdef G__MULTIBYTE
         default:
            if (G__IsDBCSLeadByte(c) && !ignoreflag) {
               string.Set(i++, c);
               c = G__fgetc();
               G__CheckDBCS2ndByte(c);
            }
            break;
#endif // G__MULTIBYTE
         // --
      }
      if (!ignoreflag) {
         string.Set(i++, c);
         G__CHECK(G__SECURE_BUFFER_SIZE, i >= G__LONGLINE, return EOF);
      }
      last_was_space = (c == ' ');
   }
   while (!flag);
   while ((i > offset) && (string[i-1] == ' ')) {
      --i;
   }
   string.Set(i, 0);
   return c;
}

//______________________________________________________________________________
int G__getstream(const char* source, int* isrc, char* string, const char* endmark)
{
   // -- Get substring of source, until one of endmark char is found.
   //
   //  char *source;      : source string. If NULL, read from input file
   //  int *isrc;         : char position of the *source if source!=NULL
   //  char *string       : output, string which is scanned from source
   //  char *endmark      : specify endmark characters
   //  
   //  
   //   Get substring of char *source; until one of endmark char is found.
   //  Return string is not used.
   //   Only used in G__getexpr() to handle 'cond?true:faulse' opeartor.
   //  
   //    char *endmark=";";
   //    char *source="abcdefg * hijklmn ;   "
   //                  ------------------^      *string="abcdefg*hijklmn"
   //  
   //    char *source="abcdefg * hijklmn) ;   "
   //                  -----------------^       *string="abcdefg*hijklmn"
   //  
   size_t i = 0;
   size_t l = 0;
   int c = 0;
   int nest = 0;
   bool single_quote = false;
   bool double_quote = false;
   bool breakflag = false;
   bool ignoreflag = false;
   bool commentflag = false;
   int start_line = G__ifile.line_number;

   do {
      ignoreflag = false;
      c = source[(*isrc)++];

      if (nest <= 0 && !single_quote && !double_quote) {
         l = 0;
         int prev;
         while ((prev = endmark[l++]) != '\0') {
            if (c == prev) {
               breakflag = true;
               ignoreflag = true;
            }
         }
      }

      bool next_single_quote = single_quote;
      bool next_double_quote = double_quote;

      switch (c) {
         case '\f':
         case '\n':
         case '\r':
         case '\t':
         case ' ':
            commentflag = false;
            if (!single_quote && !double_quote) {
               c = ' ';
            }
            break;
         case '{':
         case '(':
         case '[':
            if (!single_quote && !double_quote) {
               ++nest;
            }
            break;
         case '}':
         case ')':
         case ']':
            if (!single_quote && !double_quote) {
               --nest;
               if (nest < 0) {
                  breakflag = true;
                  ignoreflag = true;
               }
            }
            break;
         case '"':
            if (!single_quote) {
               next_double_quote = !double_quote;
            }
            break;
         case '\'':
            if (!double_quote) {
               next_single_quote = !single_quote;
            }
            break;
         case '\\':
            if (!ignoreflag) {
               string[i++] = c;
               c = source[(*isrc)++];
               if ( (c=='\n' || c=='\r') && (single_quote == 0) && (double_quote == 0)) {
                  // This is a line continuation, we just ignore both the \ and \n.
                  --i;
                  ignoreflag = 1;
               }
            }
            break;
         case '/':
            if (!double_quote && !single_quote && i > 0 && string[i-1] == '/' &&
                  commentflag) {
               --i;
               G__fignoreline();
               ignoreflag = true;
            }
            else {
               commentflag = true;
            }
            break;

         case '*':
            /* comment */
            if (!double_quote && !single_quote) {
               if (i > 0 && string[i-1] == '/' && commentflag) {
                  while ((c = source[(*isrc)++]) && c != '*' && source[*isrc] != '/') {}
                  --i;
                  ignoreflag = true;
               }
            }
            break;
         case '\0':
            breakflag = 1;
            ignoreflag = 1;
            break;
         case EOF:
            G__fprinterr(G__serr, "Error: Missing one of '%s' expected at or after line %d.\n", endmark, start_line);
            G__unexpectedEOF("G__getstream()");
            string[i] = '\0';
            break;
         // --
#ifdef G__MULTIBYTE
         default:
            if (G__IsDBCSLeadByte(c) && !ignoreflag) {
               string[i++] = c;
               c = source[(*isrc)++];
               G__CheckDBCS2ndByte(c);
            }
            break;
#endif // G__MULTIBYTE
         // --
      }
      if (!ignoreflag) {
         // i > 0, not i > offset: we care even about previous call's content of string
         if (i > 0 && !single_quote && !double_quote && string[i - 1] == ' ') {
            // We want to append c, but the trailing char is a space.
            if (c == ' ') --i; // replace ' ' by ' '
            else if (i == 1) {
               // string is " " - remove leading space.
               --i;
            } else {
               char prev = string[i - 2];
               // We only keep spaces between "identifiers" like "new const long long"
               // and between '> >'
               if ((G__IsIdentifier(prev) && G__IsIdentifier(c)) || (prev == '>' && c == '>')) {
               } else {
                  // replace previous ' '
                  --i;
               }
            }
         }
         string[i++] = c;
      }

      single_quote = next_single_quote;
      double_quote = next_double_quote;
   }
   while (!breakflag);

   if (i > 0 && string[i - 1] == ' ') --i;
   string[i] = 0;

   return c;
}

//______________________________________________________________________________
int G__fignorestream(const char* endmark)
{
   // -- Skip source file until specified endmark char appears.
   //
   //  endmark: specify endmark characters
   // 
   //  1) read source file.
   //  2) When one of endmark char appears or parenthesis nesting of
   //    parenthesis gets negative , like '())' , stop reading and
   //    return the last char.
   //
   //  Note: The file is left positioned at the next character
   //        position after the endmark character.
   // 
   //   *endmark == ";"
   //      '  ab cd e f g;hijklm '
   //       -------------^           return(';');
   // 
   //   *endmark == ")"
   //      ' abc );'
   //       -----^                   return(')');
   // 
   short l = 0;
   int c = 0;
   int prev = 0;
   short nest = 0;
   short single_quote = 0;
   short double_quote = 0;
   short flag = 0;
   int start_line = G__ifile.line_number;
   do {
      c = G__fgetc();
      if (!nest && !single_quote && !double_quote) {
         l = 0;
         while ((prev = endmark[l++])) {
            if (c == prev) {
               flag = 1;
            }
         }
      }
      switch (c) {
         case '{':
         case '(':
         case '[':
            if (!single_quote && !double_quote) {
               ++nest;
            }
            break;
         case '}':
         case ')':
         case ']':
            if (!single_quote && !double_quote) {
               --nest;
               if (nest < 0) {
                  flag = 1;
               }
            }
            break;
         case '"':
            if (!single_quote) {
               double_quote ^= 1;
            }
            break;
         case '\'':
            if (!double_quote) {
               single_quote ^= 1;
            }
            break;
         case '\\':
            // FIXME: This does *not* handle continued lines correctly!
            if (!flag) {
               c = G__fgetc();
            }
            break;
         case '/':
            // -- Check for a possible comment.
            if (!single_quote && !double_quote) {
               // Read ahead one character.
               c = G__fgetc();
               switch (c) {
                  case '*':
                     // -- C style comment.
                     G__skip_comment();
                     break;
                  case '/':
                     // -- C++ style comment.
                     G__fignoreline();
                     break;
                  default:
                     // -- Undo the readahead, backup one character.
                     fseek(G__ifile.fp, -1, SEEK_CUR);
                     // Undo the line number increment if needed.
                     if (c == '\n') {
                        --G__ifile.line_number;
                     }
                     if (G__dispsource) {
                        // -- We already printed this char, do not print it again.
                        G__disp_mask = 1;
                     }
                     // We had read a slash.
                     c = '/';
                     break;
               }
            }
            break;
         case EOF:
            G__fprinterr(G__serr, "Error: Missing one of '%s' expected at or after line %d.\n", endmark, start_line);
            G__unexpectedEOF("G__fignorestream():3");
            return c;
         // --
#ifdef G__MULTIBYTE
         default:
            if (G__IsDBCSLeadByte(c)) {
               c = G__fgetc() ;
               G__CheckDBCS2ndByte(c);
            }
            break;
#endif // G__MULTIBYTE
         //
      }
   }
   while (!flag);
   return c;
}

//______________________________________________________________________________
int G__ignorestream(const char* source, int* isrc, const char* endmark)
{
   //  char *endmark      : specify endmark characters
   // 
   //   skip source file until specified endmark char appears.
   //  This function is identical to G__fgetstream() except it does not
   //  return char *string;
   // 
   //  1) read source file.
   //  2) When one of endmark char appears or parenthesis nesting of
   //    parenthesis gets negative , like '())' , stop reading and
   //    return the last char.
   // 
   //   *endmark=";"
   //      '  ab cd e f g;hijklm '
   //       -------------^           return(';');
   // 
   //   *endmark=";"
   //      ' abc );'
   //       -----^                   return(')');
   // 
   short l;
   int c, prev;
   short nest = 0, single_quote = 0, double_quote = 0, flag = 0;
   int start_line = G__ifile.line_number;


   do {
      c = source[(*isrc)++] ;


      if ((nest == 0) && (single_quote == 0) && (double_quote == 0)) {
         l = 0;
         while ((prev = endmark[l++]) != '\0') {
            if (c == prev) {
               flag = 1;
            }
         }
      }

      switch (c) {
         case '{':
         case '(':
         case '[':
            if ((single_quote == 0) && (double_quote == 0)) {
               nest++;
            }
            break;
         case '}':
         case ')':
         case ']':
            if ((single_quote == 0) && (double_quote == 0)) {
               nest--;
               if (nest < 0) {
                  flag = 1;
               }
            }
            break;
         case '"':
            if (single_quote == 0) {
               double_quote ^= 1;
            }
            break;
         case '\'':
            if (double_quote == 0) {
               single_quote ^= 1;
            }
            break;

         case '\\':
            if (flag == 0) c = source[(*isrc)++] ;
            break;

#ifdef G__NEVER
         case '/':
            if ((single_quote == 0) && (double_quote == 0)) {
               /* comment */

               c = source[(*isrc)++] ;
               switch (c) {
                  case '*':
                     G__skip_comment();
                     break;
                  case '/':
                     G__fignoreline();
                     break;
                  default:
                     fseek(G__ifile.fp, -1, SEEK_CUR);
                     if (c == '\n' /* || c=='\r' */) --G__ifile.line_number;
                     if (G__dispsource) G__disp_mask = 1;
                     c = '/';
                     /* flag=1; BUG BUG, WHY */
                     break;
               }
            }
            break;
#endif

            /* need to handle preprocessor statements */

         case EOF:
            G__fprinterr(G__serr, "Error: Missing one of '%s' expected at or after line %d.\n", endmark, start_line);
            G__unexpectedEOF("G__fignorestream():3");
            return(c);

#ifdef G__MULTIBYTE
         default:
            if (G__IsDBCSLeadByte(c)) {
               c = source[(*isrc)++] ;
               G__CheckDBCS2ndByte(c);
            }
            break;
#endif
      }

   }
   while (flag == 0) ;

   return(c);
}

//______________________________________________________________________________
void G__fignoreline()
{
   // -- Read and ignore line (handle continuation lines as well).
   // 'as aljaf alijflijaf lisflif\n'
   // ----------------------------^
   int c;
   while ((c = G__fgetc()) != '\n' && c != '\r' && c != EOF) {
#ifdef G__MULTIBYTE
      if (G__IsDBCSLeadByte(c)) {
         c = G__fgetc();
         G__CheckDBCS2ndByte(c);
      }
      else if (c == '\\') {
         c = G__fgetc();
         if ('\r' == c || '\n' == c) c = G__fgetc();
      }
#else /* MULTIBYTE */
      if (c == '\\') {
         c = G__fgetc();
         if ('\r' == c || '\n' == c) c = G__fgetc();
      }
#endif /* MULTIBYTE */
   }
}

//______________________________________________________________________________
void G__fignoreline_peek()
{
   // -- Read and ignore a line during a peek (handle continuation lines as well).
   // 'as aljaf alijflijaf lisflif\n'
   // ----------------------------^
#ifdef G__MULTIBYTE
   int c = fgetc(G__ifile.fp);
   while ((c != EOF) && (c != '\n') && (c != '\r')) {
      if (G__IsDBCSLeadByte(c)) {
         c = fgetc(G__ifile.fp);
         G__CheckDBCS2ndByte(c);
      }
      else if (c == '\\') {
         c = fgetc(G__ifile.fp);
         if ((c == '\r') || (c == '\n')) {
            c = fgetc(G__ifile.fp);
         }
      }
      c = fgetc(G__ifile.fp);
   }
#else // MULTIBYTE
   int c = fgetc(G__ifile.fp);
   while ((c != EOF) && (c != '\n') && (c != '\r')) {
      if (c == '\\') {
         c = fgetc(G__ifile.fp);
         if ((c == '\r') || (c == '\n')) {
            c = fgetc(G__ifile.fp);
         }
      }
      c = fgetc(G__ifile.fp);
   }
#endif // MULTIBYTE
   // --
}

//______________________________________________________________________________
int G__fgetline(char* string)
{
   // --
   //   'as aljaf alijflijaf lisflif\n'
   //   ----------------------------^
   int c;
   int i = 0;
   while ((c = G__fgetc()) != '\n' && c != '\r' && c != EOF) {
      string[i] = c;
      if (c == '\\') {
         c = G__fgetc();
         if ('\r' == c || '\n' == c) c = G__fgetc();
         string[i] = c;
      }
      ++i;
   }
   string[i] = '\0';
   return(c);
}

//______________________________________________________________________________
void G__fsetcomment(G__comment_info* pcomment)
{
   //  xxxxx;            // comment      \n
   //       ^ ------------V-------------->
   int c;
   fpos_t pos;
   if (pcomment->filenum >= 0 || pcomment->p.com) return;
   fgetpos(G__ifile.fp, &pos);
   while ((isspace(c = fgetc(G__ifile.fp)) || ';' == c) && '\n' != c && '\r' != c) ;
   if ('/' == c) {
      c = fgetc(G__ifile.fp);
      if ('/' == c || '*' == c) {
         while (isspace(c = fgetc(G__ifile.fp))) {
            if ('\n' == c || '\r' == c) {
               fsetpos(G__ifile.fp, &pos);
               return;
            }
         }
         if (G__ifile.fp == G__mfp) pcomment->filenum = G__MAXFILE;
         else                    pcomment->filenum = G__ifile.filenum;
         fseek(G__ifile.fp, -1, SEEK_CUR);
         fgetpos(G__ifile.fp, &pcomment->p.pos);
      }
   }
   fsetpos(G__ifile.fp, &pos);
   return;
}

//______________________________________________________________________________
void G__set_eolcallback(void* eolcallback)
{
   // --
   G__eolcallback = (G__eolcallback_t) eolcallback;
}

//______________________________________________________________________________
int G__fgetc()
{
   // -- Read one char from source file.
   // Count line number
   // Set G__break=1 when line number comes to break point
   // Display new line number if G__dispsource==1
   // Display read character if G__dispsource==1
   //
   int c = 0;
   while (1) {
      c = fgetc(G__ifile.fp);
      switch (c) {
         case '\n':
         // case '\r':
            // -- New line char seen, move to next line number.
            ++G__ifile.line_number;
            // Check for a breakpoint, and flag a break request if needed.
            if (
               !G__nobreak &&
               !G__disp_mask &&
               G__srcfile[G__ifile.filenum].breakpoint &&
               (G__ifile.line_number < G__srcfile[G__ifile.filenum].maxline) &&
               (G__srcfile[G__ifile.filenum].breakpoint[G__ifile.line_number] |= !G__no_exec) & G__TESTBREAK &&
               !G__cintv6
            ) {
               G__BREAKfgetc();
            }
            G__eof_count = 0;
            // Display the new line number, if requested.
            if (G__dispsource) {
               G__DISPNfgetc();
            }
            // Call the end of line callback if there is one.
            if (G__eolcallback) {
               (*G__eolcallback)(G__ifile.name, G__ifile.line_number);
            }
            break;
         case EOF:
            G__EOFfgetc();
            break;
         case '\0':
            {
               // Check for end of a function-style macro.
               int was_reading_macro = G__maybe_finish_macro();
               if (was_reading_macro) {
                  // -- It was the end of a function-style macro, read next character.
                  continue;
               }
            }
            // Otherwise, fall through to the default case.
         default:
            if (G__dispsource) {
               G__DISPfgetc(c);
            }
            break;
      }
      break;
   }
   return c;
}

//______________________________________________________________________________
int G__fgetc_for_peek()
{
   // -- Read one char from source file, no semantic actions.
   //
   int c = fgetc(G__ifile.fp);
   return c;
}

/*
 * Local Variables:
 * c-tab-always-indent:nil
 * c-indent-level:3
 * c-continued-statement-offset:3
 * c-brace-offset:-3
 * c-brace-imaginary-offset:0
 * c-argdecl-indent:0
 * c-label-offset:-3
 * compile-command:"make -k"
 * End:
 */
