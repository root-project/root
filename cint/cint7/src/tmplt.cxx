/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file tmplt.c
 ************************************************************************
 * Description:
 *  Class and member function template
 ************************************************************************
 * Copyright(c) 1995~2004  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "Api.h"
#include "Dict.h"
#include "Reflex/Builder/TypeBuilder.h"
#include "common.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace Cint::Internal;
using namespace std;

#if 0
//
//  Function Map
//
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-  Static

G__IntList_new used by G__instantiate_templateclass, G__IntList_add, G__IntList_addunique
G__IntList_init used only by G__IntList_new
G__IntList_add *unused*
G__IntList_addunique used by G__instantiate_templateclass
G__IntList_delete *unused*
G__IntList_find *unused*
G__IntList_free used by G__freedeftemplateclass

G__generate_template_dict used only by G__instantiate_templateclass
   G__getIndex used only by G__generate_template_dict
   G__isSource used only by G__generate_template_dict

G__freetemplatearg used by G__resolve_specialization, G__createtemplateclass, G__declare_template, G__freedeftemplateclass, G__freetemplatefunc

G__replacetemplate used by G__instantiate_templatememfunclater, G__explicit_template_specialization, G__instantiate_templateclass, G__templatefunc, G__add_templatefunc
    G__templatesubstitute used only by G__replacetemplate

G__gettemplatearglist used by G__instantiate_templatememfunclater, G__instantiate_templateclass, G__templatefunc, G__add_templatefunc
    G__templatemaptypename used only by G__gettemplatearglist
    G__expand_def_template_arg used only by G__gettemplatearglist

G__read_specializationarg used by G__createtemplateclass, G__resolve_specialization
G__checkset_charlist used by G__matchtemplatefunc, G__add_templatefunc

G__matchtemplatefunc used by G__templatefunc, G__add_templatefunc


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-  Parsing

G__declare_template bc_parse.cxx parse.cxx
    G__read_formal_templatearg used only by G__declare_template
    G__explicit_template_specialization used only by G__declare_template
    G__createtemplateclass struct.cxx(G__set_class_autoloading_table) newlink.cxx(G__tagtable_setup), G__declare_template
        G__instantiate_templateclasslater used only by G__createtemplateclass
    G__createtemplatefunc used only by G__declare_template
        G__istemplatearg used only by G__createtemplatefunc
    G__createtemplatememfunc used only by G__declare_template
        G__instantiate_templatememfunclater used only by G__createtemplatememfunc


G__freedeftemplateclass scrupto.cxx
    G__freetemplatememfunc used only by G__freedeftemplateclass

G__freetemplatefunc scrupto.cxx

G__instantiate_templateclass struct.cxx(G__defined_tagname{implicit instantiation}) parse.cxx(G__keyword_anytime_8{explicit instantiation}), G__instantiate_templateclasslater
    G__cattemplatearg used only by G__instantiate_templateclass
    G__settemplatealias used only by G__instantiate_templateclass
    G__resolve_specialization used only by G__instantiate_templateclass
        G__modify_callpara used only by G__resolve_specialization
            G__delete_string used only by G__modify_callpara
            G__delete_end_string used only by G__modify_callpara

G__templatefunc ifunc.cxx func.cxx
G__add_templatefunc ifunc.cxx, newlink.cxx


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-  Display

G__gettemplatearg disp.cxx


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-  Searching

G__defined_templateclass struct.cxx macro.cxx bc_parse.cxx expr.cxx fread.cxx ifunc.cxx parse.cxx func.cxx disp.cxx newlink.cxx decl.cxx, G__createtemplatememfunc, G__matchtemplatefunc, G__templatefunc, G__createtemplatefunc

G__defined_templatefunc expr.cxx ifunc.cxx func.cxx, G__defined_templatememfunc, G__add_templatefunc

G__defined_templatememfunc expr.cxx
    G__getobjecttagnum used only by G__defined_templatememfunc

#endif // 0


// Static Functions
static G__IntList* G__IntList_new(long iin, G__IntList* prev);
static void G__IntList_init(G__IntList* body, long iin, G__IntList* prev);
static void G__IntList_addunique(G__IntList* body, long iin);
static void G__IntList_free(G__IntList* body);

static int G__generate_template_dict(const char* template_id, G__Definedtemplateclass* class_tmpl, G__Charlist* tmpl_arg_list);
static int G__getIndex(int index,::Reflex::Type tagnum, std::vector<std::string>& headers);
static bool G__isSource(const char* filename);

static void G__freetemplatearg(G__Templatearg* def_para);
static void G__replacetemplate(char* templatename, const char* tagname, G__Charlist* callpara, FILE* def_fp, int line, int filenum, fpos_t* pdef_pos, G__Templatearg* def_para, int isclasstemplate, int npara, int parent_tagnum);
static int G__templatesubstitute(char* symbol, G__Charlist* callpara, G__Templatearg* defpara, char* templatename, const char* tagname, int c, int npara, int isnew);
//858//static int G__gettemplatearglist(char* tmpl_arg_string, G__Charlist* tmpl_arg_list_in, G__Templatearg* tmpl_para_list, int* pnpara, int parent_tagnum, G__Definedtemplateclass* specialization);
static int G__gettemplatearglist(char* tmpl_arg_string, G__Charlist* tmpl_arg_list_in, G__Templatearg* tmpl_para_list, int* pnpara, int parent_tagnum);
static void G__templatemaptypename(char* string);
static char* G__expand_def_template_arg(char* str_in, G__Templatearg* def_para, G__Charlist* charlist);
static G__Templatearg* G__read_specializationarg(G__Templatearg* tmpl_params, char* source);
static int G__checkset_charlist(char* type_name, G__Charlist* pcall_para, int narg, int ftype);
static int G__matchtemplatefunc(G__Definetemplatefunc* deftmpfunc, G__param* libp, G__Charlist* pcall_para, int funcmatch);

static G__Templatearg* G__read_formal_templatearg();
static void G__explicit_template_specialization();
static void G__instantiate_templateclasslater(G__Definedtemplateclass* deftmpclass);
static int G__createtemplatefunc(char* funcname, G__Templatearg* targ, int line_number, fpos_t* ppos);
static int G__istemplatearg(char* paraname, G__Templatearg* def_para);
static int G__createtemplatememfunc(char* new_name);
static void G__instantiate_templatememfunclater(G__Definedtemplateclass* deftmpclass, G__Definedtemplatememfunc* deftmpmemfunc);

static void G__freetemplatememfunc(G__Definedtemplatememfunc* memfunctmplt);

static void G__cattemplatearg(char* template_id, G__Charlist* tmpl_arg_list, int npara = 0);
static void G__settemplatealias(const char* scope_name, const char* template_id, int tagnum, G__Charlist* tmpl_arg_list, G__Templatearg* tmpl_param, int create_in_envtagnum);
static G__Definedtemplateclass* G__resolve_specialization(char* tmpl_args_string, G__Definedtemplateclass* class_tmpl, G__Charlist* expanded_tmpl_arg_list);
static void G__modify_callpara(G__Templatearg* spec_arg_in, G__Templatearg* given_spec_arg_in, G__Charlist* expanded_tmpl_arg_in);
static void G__delete_string(char* str, const char* del);
static void G__delete_end_string(char* str, const char* del);

static ::Reflex::Type G__getobjecttagnum(char *name);


// Cint Internal Functions
namespace Cint {
namespace Internal {
void G__declare_template();
int G__createtemplateclass(char* new_name, G__Templatearg* targ, int isforwarddecl);
void G__freedeftemplateclass(G__Definedtemplateclass* deftmpclass);
void G__freetemplatefunc(G__Definetemplatefunc* deftmpfunc);
int G__instantiate_templateclass(char* tagnamein, int noerror);
int G__templatefunc(G__value* result, const char* funcname, G__param* libp, int hash, int funcmatch);
G__funclist* G__add_templatefunc(const char* funcnamein, G__param* libp, int hash, G__funclist* funclist, const ::Reflex::Scope p_ifunc, int isrecursive);
char* G__gettemplatearg(int n, G__Templatearg* def_para);
G__Definetemplatefunc* G__defined_templatefunc(const char* name);
G__Definetemplatefunc* G__defined_templatememfunc(const char* name);
}
}

// Functions in the C interface.
extern "C" G__Definedtemplateclass* G__defined_templateclass(const char* name);


static int G__templatearg_enclosedscope = 0; // Used only to determine parent_tagnum parameter to G__settemplatealias.

//______________________________________________________________________________
//______________________________________________________________________________
//
//  G__IntList Functions.  Used for list of instantiated tagnums.
//

//______________________________________________________________________________
static G__IntList* G__IntList_new(long iin, G__IntList* prev)
{
   G__IntList* body = new G__IntList;
   G__IntList_init(body, iin, prev);
   return body;
}

//______________________________________________________________________________
static void G__IntList_init(G__IntList* body, long iin, G__IntList* prev)
{
   body->i = iin;
   body->next = 0;
   body->prev = prev;
}

//______________________________________________________________________________
static void G__IntList_addunique(G__IntList* body, long iin)
{
   while (body->next) {
      if (body->i == iin) {
         return;
      }
      body = body->next;
   }
   if (body->i == iin) {
      return;
   }
   body->next = G__IntList_new(iin, body);
}

//______________________________________________________________________________
static void G__IntList_free(G__IntList* body)
{
   if (!body) {
      return;
   }
   if (body->prev) {
      body->prev->next = 0;
   }
   while (body->next) {
      G__IntList_free(body->next);
   }
   delete body;
}

//______________________________________________________________________________
//______________________________________________________________________________

//______________________________________________________________________________
static int G__generate_template_dict(const char* template_id, G__Definedtemplateclass* class_tmpl, G__Charlist* tmpl_arg_list)
{
   // Generate a dictionary for template_id using G__GenerateDictionary().
   // Go through all header files and collect only the ones needed
   // for the specified template class.
   //
   Cint::G__pGenerateDictionary pGD = Cint::G__GetGenerateDictionary();
   if (!pGD) { // Dictionary generation is disabled.
      return -1;
   }
   if (G__def_tagnum && !G__def_tagnum.IsTopScope()) { // Ignore member templates.
      return -1;
   }
   vector<string> headers; // List of header files.
   //
   // Add header file for definition of the template-id to the list.
   //
   int fileNum = class_tmpl->filenum;
   if (fileNum < 0) {
      return -1;
   }
   fileNum = G__getIndex(fileNum, Reflex::Type(), headers);
   if (fileNum == -1) {
      return -1;
   }
   //
   //  Add any header files for the definitions of template
   //  arguments of class type to the list.
   //
   for ( ; tmpl_arg_list->next; tmpl_arg_list = tmpl_arg_list->next) {
      G__value gValue = G__string2type_body(tmpl_arg_list->string, 1);
      ::Reflex::Type ty = G__value_typenum(gValue).RawType();
      if (ty.IsClass()) { // FIXME: We need to support union here too!
         int index = G__getIndex(-1, ty, headers); // Note: Do real work, headers is modified here.
         if (index == -1) {
            return -1;
         }
      }
   }
   //
   //  Call ACLiC to generate and load the dictionary.
   //
   string template_id_str = template_id;
   int rtn = pGD(template_id_str, headers);
   if (rtn) {
      return -1;
   }
   //
   //  Now lookup the passed template-id.
   //
   int tagnum = G__defined_tagname(template_id_str.c_str(), 3); 
   //
   //  If dictionary generation succeeded, remember what
   //  header files we used in the comment string for
   //  the generated template instantiation.
   //
   if (tagnum != -1) {
      G__RflxProperties *prop = G__get_properties(G__Dict::GetDict().GetType(tagnum));
      if (!prop->comment.p.com) {
         string headersToInclude("//[INCLUDE:");
         for (vector<string>::iterator iter = headers.begin(); iter != headers.end(); ++iter) {
            headersToInclude += *iter + ";";
         }
         prop->comment.p.com = new char[headersToInclude.size() + 1];
         strcpy(prop->comment.p.com, headersToInclude.c_str());
      }
   }
   return tagnum;
}

//______________________________________________________________________________
static int G__getIndex(int index, Reflex::Type tagnum, std::vector<std::string>& headers)
{
   // Find the header file or shared library which contains the definition
   // of the passed template instantiation.  If no instantiation is passed,
   // then the nearest including header file or shared libary file is found,
   // which is usually the passed file itself.
   //
   // If a header file is found, returns the index in G__srcfile of the found
   // header file, and appends the header filename to headers if it is not
   // already in the list.
   //
   // If a shared library file is found, the passed template instantiation
   // is checked for a comment which includes the string "//[INCLUDE:".
   // If found, the ";" separated list of filenames is appended to headers,
   // and the index in G__srcfile of the found shared library file is returned.
   //
   //--
   //
   //  Walk the inclusion chain backwards starting from the given
   //  index looking for the index of the last file included from
   //  a shared library or a source file (not a header file).
   //
   G__RflxProperties *prop = 0;
   if (tagnum) {
      prop = G__get_properties(tagnum);
      if (index==-1) {
         index = prop->filenum;
         if (index < 0) {
            return -1;
         }
         if (G__srcfile[index].filename[0] == '{') {
            // ignore "{CINTEX dictionary translator}"
            return -1;
         }
      }
   } else if (index == -1) {
      return -1;
   }
   for (
      ;
      (G__srcfile[index].included_from > -1) && (G__srcfile[index].included_from < G__nfile);
      index = G__srcfile[index].included_from
   ) {
      const int inc_from = G__srcfile[index].included_from;
      if (
         (G__srcfile[inc_from].slindex != -1) || // included from is a shared library, or
         G__isSource(G__srcfile[inc_from].filename) // included from is a source file (not a header file)
      ) {
         break;
      }
   }
   if (G__srcfile[index].slindex == -1) { // We found a header file.
      // Add found file to list if it is not already there, and return its index.
      vector<string>::iterator iter = find(headers.begin(), headers.end(), G__srcfile[index].filename);
      if (iter == headers.end()) {
         headers.push_back(G__srcfile[index].filename);
      }
      return index;
   }
   //
   //  At this point, we have a shared library.
   //
   if (!tagnum) { // No template instantiation given, return error.
      return -1;
   }
   //
   //  FIXME: This block of code does not process the include string
   //  FIXME: correctly, apparently the filename list is meant to be
   //  FIXME: separated by commas, but we do not advance the pointer
   //  FIXME: to the next filename after finding the first one.
   //
   if ( // The template instantiation has our special comment.
      prop &&
      prop->comment.p.com &&
      strstr(prop->comment.p.com, "//[INCLUDE:")
   ) { // The template instantiation has our special comment.
      char* p = prop->comment.p.com;
      while (*p && (*p != ':')) {
         ++p;
      }
      if (*p) {
         ++p;
      }
      // If so, add all headers from G__struct..comment.p.com to headers.push_back(...) and go on.
      string tmpHeader;
      for ( ; *p; ++p) {
         if ((*p != ';')) {
            tmpHeader += *p;
         }
         else {
            vector<string>::iterator iter = find(headers.begin(), headers.end(), tmpHeader);
            if (iter == headers.end()) {
               headers.push_back(tmpHeader);
            }
            tmpHeader.clear();
         }
      }
      return index;
   }
   //
   //  Return error, we found no suitable files.
   //
   return -1;
}

//______________________________________________________________________________
static bool G__isSource(const char* filename)
{
   // Very simple check (by extension),if the file is a source file.
   const char* ptr = strrchr(filename, '.');
   if (ptr && ((ptr[1] == 'c') || (ptr[1] == 'C'))) {
      return true;
   }
   return false;
}

//______________________________________________________________________________
//______________________________________________________________________________

//______________________________________________________________________________
static void G__freetemplatearg(G__Templatearg* def_para)
{
   // Free a defined template parameter list.
   if (def_para) {
      if (def_para->next) {
         G__freetemplatearg(def_para->next);
      }
      if (def_para->string) {
         free(def_para->string);
      }
      if (def_para->default_parameter) {
         free(def_para->default_parameter);
      }
      delete def_para;
   }
}

//______________________________________________________________________________
#define SET_READINGFILE \
   fgetpos(G__mfp,&out_pos); \
   fsetpos(G__ifile.fp,&in_pos)

//______________________________________________________________________________
#define SET_WRITINGFILE \
   fgetpos(G__ifile.fp,&in_pos); \
   fsetpos(G__mfp,&out_pos)

//______________________________________________________________________________
static void G__replacetemplate(char* templatename, const char* tagname, G__Charlist* tmpl_arg_list, FILE* def_fp, int line, int filenum, fpos_t* pdef_pos, G__Templatearg* tmpl_para_list, int isclasstemplate, int npara, int parent_tagnum)
{
   // Replace template string and prerun
   //
   //  npara:
   //
   //       Limit on number of template arguments to output
   //       in a template-id for our template.  This allows
   //       us to omit default template arguments in the
   //       source code we write to the output file.
   //
   fpos_t store_mfpos;
   int store_mfline;
   fpos_t orig_pos;
   fpos_t pos;
   int c;
   int c2;
   int mparen;
   G__StrBuf symbol_sb(G__LONGLINE);
   char* symbol = symbol_sb;
   int double_quote = 0;
   int single_quote = 0;
   G__input_file store_ifile;
   int store_prerun;
   ::Reflex::Scope store_tagnum;
   ::Reflex::Scope store_def_tagnum;
   ::Reflex::Scope store_tmplt_def_tagnum;
   ::Reflex::Scope store_tagdefining;
   int store_def_struct_member;
   int store_var_type;
   int store_breaksignal;
   int store_no_exec_compile;
   int store_asm_noverflow;
   ::Reflex::Member store_func_now;
   int store_decl;
   int store_asm_wholefunction;
   int store_reftype;
   ::Reflex::Scope store_ifunc;
   int slash = 0;
   fpos_t out_pos;
   fpos_t in_pos;
   fpos_t const_pos;
   char const_c = 0;
   ::Reflex::Scope store_memberfunc_tagnum;
   int store_globalcomp;
   //
   // open macro and template substitution file and get ready for
   // template instantiation
   //
   //--
   //
   // store restart position, used later in this function
   //
   if (G__ifile.fp) {
      fgetpos(G__ifile.fp, &orig_pos);
   }
   //
   //  get tmpfile file pointer
   //
   if (!G__mfp) {
      G__openmfp();
      fgetpos(G__mfp, &G__nextmacro);
      G__mline = 1;
      store_mfline = 0;
   }
   else {
      fgetpos(G__mfp, &store_mfpos);
      store_mfline = G__mline;
      fsetpos(G__mfp, &G__nextmacro);
   }
   //
   //  Note start of process.
   //
   if (G__dispsource) {
      G__fprinterr(G__serr, "\n!!!Instantiating template %s\n", tagname);
   }
   //
   //  print out header
   //
   ++G__mline;
   fprintf(G__mfp, "// template %s  FILE:%s LINE:%d\n", tagname , G__ifile.name, G__ifile.line_number);
   if (G__dispsource) {
      G__fprinterr(G__serr, "// template %s  FILE:%s LINE:%d\n", tagname , G__ifile.name, G__ifile.line_number);
   }
   fgetpos(G__mfp, &pos);
   //
   //  Setup template definition input file pointer and position.
   //
   store_ifile = G__ifile;
   G__ifile.fp = def_fp;
   G__ifile.line_number = line;
   G__ifile.filenum = filenum;
   in_pos = *pdef_pos;
   //
   //  Note the current template source file position.
   //
   ++G__mline;
   fprintf(G__mfp, "# %d \"%s\"\n", G__ifile.line_number, G__srcfile[G__ifile.filenum].filename);
   if (G__dispsource) {
      G__fprinterr(G__serr, "# %d \"%s\"\n", G__ifile.line_number, G__srcfile[G__ifile.filenum].filename);
   }
   //
   // We are always ignoring the :: when they are alone (and thus specify
   // the global name space, we also need to ignore them here!
   //
   if (!strncmp(templatename, "::", 2)) {
      templatename += 2;
   }
   if (!strncmp(tagname, "::", 2)) {
      tagname += 2;
   }
   //
   //  Read template definition from file and substitute the template arguments.
   //
   int isnew = 0;
   const char* punctuation = "\t\n !\"#$%&'()*+,-./:;<=>?@[\\]^{|}~"; // not alpha, numeric, underscore, CR, FF, VT, backtick
   mparen = 0;
   while (1) { // loop over all punctuation and whitespace separated words in definition
      G__disp_mask = 10000;
      SET_READINGFILE;
      c = G__fgetstream(symbol, punctuation);
      SET_WRITINGFILE;
      //fprintf(stderr, "read symbol: '%s' stopped at char: '%c'\n", symbol, c);
      if (c == '~') { // Word is terminated by a tilde
         isnew = 1;
      }
      else if (c == ',') {
         isnew = 0; // Reset isnew state at last word before a comma.
      }
      else if (c == ';') {
         isnew = 0; // Reset isnew state at last word before a semicolon.
         const_c = 0;
      }
      //
      //  Handle a word.
      //
      if (symbol[0]) { // We got a word between punctuation characters or whitespace.
         if (!double_quote && !single_quote) { // not in a quoted string
            //
            //  If separator was whitespace, then skip all
            //  following whitespace to the end of the line.
            //
            if (isspace(c)) { // Separator was whitespace, skip following whitespace.
               c2 = c;
               SET_READINGFILE;
               while (isspace(c = G__fgetc())) {
                  if (c == '\n') {
                     break;
                  }
               }
               if (c != '<') {
                  fseek(G__ifile.fp, -1, SEEK_CUR);
                  c = c2;
               }
               SET_WRITINGFILE;
            }
            if (!strcmp("new", symbol)) { // Word is "new", set the "new" seen flag.
               isnew = 1; // Remember we have seen "new"
            }
            else if (!strcmp("operator", symbol)) { // Special case the word "operator", we may have "<" and ">" to confuse our parse.
               SET_READINGFILE;
               if (c == '(') {
                  // operator() ()
                  size_t len = strlen(symbol);
                  symbol[len + 1] = 0;
                  symbol[len] = '(';
                  ++len;
                  c = G__fgetstream(symbol + len, ")"); // add '('
                  len = strlen(symbol);
                  symbol[len + 1] = 0;
                  symbol[len] = ')';
                  ++len;
                  c = G__fgetstream(symbol + len, punctuation); // add ')'
               }
               else if (c == '<') {
                  // operator <, <=, <<
                  size_t len = strlen(symbol);
                  symbol[len + 1] = 0;
                  symbol[len] = '<';
                  ++len;
                  c = G__fgetc();
                  if ((c == '<') || (c == '=')) {
                     symbol[len + 1] = 0;
                     symbol[len] = c;
                     c = G__fgetc();
                  }
               }
               else {
                  size_t len = strlen(symbol);
                  size_t templsubst_upto = 8;
                  do {
                     symbol[len + 1] = 0;
                     symbol[len] = c;
                     ++len;
                     c = G__fgetc();
                     // replace T of "operator T const*"
                     if (c && ((c == ' ') || strchr(punctuation, c))) {
                        int ret = G__templatesubstitute(symbol + templsubst_upto + 1, tmpl_arg_list, tmpl_para_list, templatename, tagname, c, npara, 1);
                        //
                        //  If we have reached the npara limit in a template-id
                        //  matching our template, shorten the template-id
                        //  in the output file.
                        //
                        if (ret && (c != '>')) {
                           G__StrBuf ignorebuf_sb(G__LONGLINE);
                           char* ignorebuf = ignorebuf_sb;
                           c = G__fgetstream(ignorebuf, ">");
                           G__ASSERT('>' == c);
                           c = '>';
                        }
                        len = strlen(symbol);
                        templsubst_upto = len;
                     }
                  }
                  while ((c != '(') && (c != '<')); // deficiency: no conversion to templated class
                  // replace T of "operator const T"
                  if ((symbol[templsubst_upto] == ' ') || strchr(punctuation, symbol[templsubst_upto])) {
                     int ret = G__templatesubstitute(symbol + templsubst_upto + 1, tmpl_arg_list, tmpl_para_list, templatename, tagname, c, npara, 1);
                     //
                     //  If we have reached the npara limit in a template-id
                     //  matching our template, shorten the template-id
                     //  in the output file.
                     //
                     if (ret && (c != '>')) {
                        G__StrBuf ignorebuf_sb(G__LONGLINE);
                        char* ignorebuf = ignorebuf_sb;
                        c = G__fgetstream(ignorebuf, ">");
                        G__ASSERT(c == '>');
                        c = '>';
                     }
                  }
               }
               SET_WRITINGFILE;
               isnew = 1;
            }
            //
            //  Do the template argument substitution.
            //
            int ret = G__templatesubstitute(symbol, tmpl_arg_list, tmpl_para_list, templatename, tagname, c, npara, isnew);
            //
            //  If we have reached the npara limit in a template-id
            //  matching our template, shorten the template-id
            //  in the output file.
            //
            if (ret && (c != '>')) { // We haved reached the npara limit.
               G__StrBuf ignorebuf_sb(G__LONGLINE);
               char* ignorebuf = ignorebuf_sb;
               SET_READINGFILE;
               c = G__fgetstream(ignorebuf, ">");
               SET_WRITINGFILE;
               G__ASSERT(c == '>');
               c = '>';
            }
         }
         // FIXME: We should not do the const reversal in a quoted string!
         //
         //  Output word, which may have been substituted above,
         //  but check if we need to reverse the position of a
         //  "const" keyword while doing this.
         //
         if (const_c && symbol[strlen(symbol)-1] == '*') { // The type substituted was a pointer type.
            //
            //  To preserve the meaning of:
            //
            //       typedef int* MyType;
            //       const MyType p;
            //
            //  After the symbol substitution we have:
            //
            //       const int* p;
            //
            //  So we must move the const so that we have:
            //
            //       int* const p;
            //
            //  which restores the intended meaning.
            //
            fsetpos(G__mfp, &const_pos);
            fprintf(G__mfp, "%s", symbol);
            //
            //  Handle any requested tracing.
            //
            if (G__dispsource) {
               G__fprinterr(G__serr, "%s", symbol);
            }
            fprintf(G__mfp, " const%c", const_c); // Reverse the position of the const.
            //
            //  Handle any requested tracing.
            //
            if (G__dispsource) {
               G__fprinterr(G__serr, " const%c", const_c);
            }
            const_c = 0; // Reset "const" word seen state.
         }
         else if (const_c && (strstr(symbol, "*const") || strstr(symbol, "* const"))) { // The substituted type was a const pointer, we have duplicated const qualifiers, remove the duplicate.
            //
            //  Given:
            //
            //       typedef int* const MyType;
            //       const MyType p;
            //
            //  After the symbol substitution we have:
            //
            //       const int* const p;
            //
            //  So we move the first const so that we have:
            //
            //       int* const const p;
            //
            //  and then we drop the extra const.  This preserves
            //  the intended meaning.
            //
            fsetpos(G__mfp, &const_pos);
            fprintf(G__mfp, "%s", symbol); // Remove the duplicate const.
            //
            //  Handle any requested tracing.
            //
            if (G__dispsource) {
               G__fprinterr(G__serr, "%s", symbol);
            }
            fprintf(G__mfp, "%c", const_c); // printing %c is not perfect
            //
            //  Handle any requested tracing.
            //
            if (G__dispsource) {
               G__fprinterr(G__serr, "%c", const_c);
            }
            const_c = 0; // Reset "const" word seen state.
         }
         else { // No special "const" trickery to do
            const_c = 0; // Initialize "const" reversal flag.
            if ((c != ';') && !strcmp("const", symbol)) { // Not last word and word is "const".
               const_c = c; // Set "const" word seen state, and remember following punctuation char, we may move the const later.
               fgetpos(G__mfp, &const_pos); // Remember output file position where we write the word for later, we may rewrite
            }
            fprintf(G__mfp, "%s", symbol); // Output word, it may have been substituted with a template argument above.
            //
            //  Handle any requested tracing.
            //
            if (G__dispsource) {
               G__fprinterr(G__serr, "%s", symbol);
            }
         }
#if 0
         fprintf(G__mfp, "%s", symbol); // Output word, it may have been substituted with a template argument above.
         //
         //  Handle any requested tracing.
         //
         if (G__dispsource) {
            G__fprinterr(G__serr, "%s", symbol);
         }
#endif // 0
         // --
      }
      //
      //  Now handle the whitespace or punctuation
      //  separator character.
      //
      //--
      //
      //  Handle any comment lines.
      //
      if (slash == 1) { // We may have a comment.
         slash = 0; // Reset possible comment state.
         if ((c == '/') && !symbol[0] && !single_quote && !double_quote) { // Beginning of C++ style comment.
            //
            //  Flush rest of line to output file unchanged.
            //
            SET_READINGFILE;
            G__fgetline(symbol);
            SET_WRITINGFILE;
            fprintf(G__mfp, "/%s\n", symbol);
            //
            //  Do tracing if requested.
            //
            if (G__dispsource) {
               G__fprinterr(G__serr, "/%s\n", symbol);
            }
            ++G__mline; // Keep track of line number.
            continue; // next word
         }
         else if ((c == '*') && !symbol[0] && !single_quote && !double_quote) { // Beginning of C style comment.
            //
            //  Flush rest of comment to output file and
            //  continue with next word.
            //
            fprintf(G__mfp, "/\n");
            if (G__dispsource) {
               G__fprinterr(G__serr, "/\n");
            }
            ++G__mline;
            SET_READINGFILE;
            G__skip_comment();
            SET_WRITINGFILE;
            continue;
         }
      }
      //
      //  Keep track of curly brace nesting
      //  and check for termination of definition.
      //
      if (!single_quote && !double_quote) {
         if (c == '{') {
            ++mparen;
         }
         else if (c == '}') {
            --mparen;
            //
            //  Check for termination of definition.
            //
            if (!mparen) { // Definition has terminated, we are done.
               fputc(c, G__mfp); // Output closing curly brace.
               //
               //  Handle any tracing requested.
               //
#ifndef G__OLDIMPLEMENTATION1485
               if (G__dispsource) {
                  G__fputerr(c);
               }
#else // G__OLDIMPLEMENTATION1485
               if (G__dispsource) {
                  fputc(c, G__serr);
               }
#endif // G__OLDIMPLEMENTATION1485
               break; // And we are done.
            }
         }
         else if ((c == ';') && !mparen) { // Semicolon at outer level means end of declaration, we are done.
            break;
         }
      }
      //
      //  Change quoting state if necessary.
      //
      if ((c == '\'') && !double_quote) {
         single_quote = single_quote ^ 1;
      }
      else if ((c == '"') && !single_quote) {
         double_quote = double_quote ^ 1;
      }
      //
      //  Change slash seen state if necessary.
      //  We use this to check for comments.
      //
      if (c == '/') { // Possible beginning of comment.
         slash = 1; // Remember possible beginning of comment.
      }
      //
      //  If the substituted symbol ends with '>' and
      //  the separator character is '>', then insert
      //  a space to prevent creating a ">>" token.
      //
      if (symbol[0] && (symbol[strlen(symbol)-1] == '>') && (c == '>')) {
         fputc(' ', G__mfp);
         if (G__dispsource) {
            G__fprinterr(G__serr, " ");
         }
      }
      //
      //  Output the separator char.
      //
      fputc(c, G__mfp);
      //
      //  Handle any tracing requested.
      //
#ifndef G__OLDIMPLEMENTATION1485
      if (G__dispsource) {
         G__fputerr(c);
      }
#else // G__OLDIMPLEMENTATION1485
      if (G__dispsource) {
         fputc(c, G__serr);
      }
#endif // G__OLDIMPLEMENTATION1485
      //
      //  Keep track of the line number.
      //
      if ((c == '\n') || (c == '\r')) {
         ++G__mline;
      }
   }
   //
   //  And write out ending to temp file.
   //
   if (isclasstemplate == 2) { // is a forward declaration
      fprintf(G__mfp, ";");
      if (G__dispsource) {
         G__fprinterr(G__serr, ";");
      }
   }
   else if ((isclasstemplate == 1) && (c != ';')) { // is not forward declaration
      SET_READINGFILE;
      G__fgetstream(symbol, ";");
      const_c = 0;
      SET_WRITINGFILE;
      fprintf(G__mfp, "%s ;", symbol);
      if (G__dispsource) {
         G__fprinterr(G__serr, "%s ;", symbol);
      }
   }
   else if (c == ';') {
      fputc(c, G__mfp);
#ifndef G__OLDIMPLEMENTATION1485
      if (G__dispsource) {
         G__fputerr(c);
      }
#else // G__OLDIMPLEMENTATION1485
      if (G__dispsource) {
         fputc(c, G__serr);
      }
#endif // G__OLDIMPLEMENTATION1485
      // --
   }
   fputc('\n', G__mfp);
#ifndef G__OLDIMPLEMENTATION1485
   if (G__dispsource) {
      G__fputerr('\n');
   }
#else // G__OLDIMPLEMENTATION1485
   if (G__dispsource) {
      fputc('\n', G__serr);
   }
#endif // G__OLDIMPLEMENTATION1485
   ++G__mline;
   //
   //  Finish string substitution.
   //
   G__disp_mask = 0;
   fgetpos(G__mfp, &G__nextmacro);
   fflush(G__mfp);
   //
   //  Rewind tmpfile and echo it.
   //
#if 0
   if (G__dispsource) {
      G__fprinterr(G__serr, "!!! Dumping generated template instantiation %s\n", tagname);
   }
   fsetpos(G__mfp, &pos);
   {
      char mybuf[2048];
      char* p = fgets(mybuf, 2048, G__mfp);
      while (p) {
         fprintf(stderr, "%s", p);
         p = fgets(mybuf, 2048, G__mfp);
      }
   }
#endif // 0
   //
   //  Rewind tmpfile and parse template class or function.
   //
   if (G__dispsource) {
      G__fprinterr(G__serr, "!!! Reading template %s\n", tagname);
   }
   fsetpos(G__mfp, &pos);
   G__ifile.fp = G__mfp;
   store_prerun = G__prerun;
   store_tagnum = G__tagnum;
   store_def_tagnum = G__def_tagnum;
   store_tagdefining = G__tagdefining;
   store_tmplt_def_tagnum = G__tmplt_def_tagnum;
   store_def_struct_member = G__def_struct_member;
   store_var_type = G__var_type;
   store_breaksignal = G__breaksignal;
   store_no_exec_compile = G__no_exec_compile;
   store_asm_noverflow = G__asm_noverflow;
   store_func_now = G__func_now;
   store_decl = G__decl;
   store_ifunc = G__p_ifunc;
   store_asm_wholefunction = G__asm_wholefunction;
   store_reftype = G__reftype;
   store_memberfunc_tagnum = G__memberfunc_tagnum;
   store_globalcomp = G__globalcomp;
   G__prerun = 1;
   G__tagnum = ::Reflex::Scope();
   G__tmplt_def_tagnum = G__def_tagnum;
   if (G__tmplt_def_tagnum.IsTopScope()) {
      G__tmplt_def_tagnum = Reflex::Dummy::Scope();
   }
   // instantiated template objects in scope that template is declared
   G__def_tagnum = G__Dict::GetDict().GetScope(parent_tagnum);
   G__tagdefining = G__Dict::GetDict().GetScope(parent_tagnum);
   G__def_struct_member = (parent_tagnum != -1);
   if (G__exec_memberfunc) {
      G__memberfunc_tagnum = G__Dict::GetDict().GetScope(parent_tagnum);
   }
   G__var_type = 'p';
   G__breaksignal = 0;
   G__abortbytecode(); // This has to be 'suspend', indeed.
   G__no_exec_compile = 0;
   G__func_now = ::Reflex::Member();
   G__decl = 0;
   G__p_ifunc = ::Reflex::Scope(); // &G__ifunc; this old code means add to global scope ... so this seems to be already taking care of by the G__tagnum = -1 above
   G__asm_wholefunction = 0;
   G__reftype = G__PARANORMAL;
   //
   //  Parse the temporary file.
   //
   int brace_level = 0;
   G__exec_statement(&brace_level);
   //
   //
   //
   G__func_now = store_func_now;
   G__decl = store_decl;
   G__ASSERT(0 == G__decl || 1 == G__decl);
   G__p_ifunc = store_ifunc;
   G__asm_noverflow = store_asm_noverflow;
   G__no_exec_compile = store_no_exec_compile;
   G__prerun = store_prerun;
   G__tagnum = store_tagnum;
   G__tmplt_def_tagnum = store_tmplt_def_tagnum;
   G__def_tagnum = store_def_tagnum;
   G__tagdefining = store_tagdefining;
   G__def_struct_member = store_def_struct_member;
   G__var_type = store_var_type;
   G__breaksignal = store_breaksignal;
   G__asm_wholefunction = store_asm_wholefunction;
   G__reftype = store_reftype;
   G__memberfunc_tagnum = store_memberfunc_tagnum;
   G__globalcomp = store_globalcomp;
   // restore input file
   G__ifile = store_ifile;
   if (G__ifile.filenum >= 0) {
      G__security = G__srcfile[G__ifile.filenum].security;
   }
   else {
      G__security = G__SECURE_LEVEL0;
   }
   // protect the case when template is instantiated from command line
   if (G__ifile.fp) {
      fsetpos(G__ifile.fp, &orig_pos);
   }
   if (G__dispsource) {
      G__fprinterr(G__serr, "\n!!!Complete instantiating template %s\n", tagname);
   }
   if (store_mfline) {
      fsetpos(G__mfp, &store_mfpos);
   }
}

#undef SET_WRITINGFILE
#undef SET_READINGFILE

//______________________________________________________________________________
static int G__templatesubstitute(char* symbol, G__Charlist* tmpl_arg_list, G__Templatearg* tmpl_para_list, char* templatename, const char* tagname, int c, int npara, int isnew)
{
   // If given symbol matches a template parameter, substitute it with the matching given argument.
   static int state = 0; // Controls our result, flags that we are in a template-id and counts params processed.
   // template name substitution
   if (!strcmp(symbol, templatename)) { // Matched our template name, substitute if not start of template-id, change state, and return.
      state = 0;
      if (c == '<') {
         state = 1; // Flag that we are in our template-id.
      }
      else {
         strcpy(symbol, tagname);
      }
      return 0;
   }
   int result = 0;
   for ( ; tmpl_para_list; tmpl_para_list = tmpl_para_list->next) {
      if (!strcmp(symbol, tmpl_para_list->string)) { // Match with template parameter, do ..., and exit loop
         //
         //  Substitute symbol with template argument
         //  or with default value.
         //
         if (tmpl_arg_list && tmpl_arg_list->string) {
            strcpy(symbol, tmpl_arg_list->string);
         }
         else if (tmpl_para_list->default_parameter) {
            strcpy(symbol, tmpl_para_list->default_parameter);
         }
         else {
            G__fprinterr(G__serr, "Error: template argument for %s missing", tmpl_para_list->string);
            G__genericerror(0);
         }
         //
         //  TODO: What does this block do, and why?
         //
         if (
            (c == '(') &&
            symbol[0] &&
            !isnew &&
            (
               (symbol[strlen(symbol)-1] == '*') ||
               strchr(symbol, ' ') ||
               strchr(symbol, '<')
            )
         ) {
            G__StrBuf temp_sb(G__LONGLINE);
            char* temp = temp_sb;
            strcpy(temp, symbol);
            sprintf(symbol, "(%s)", temp);
         }
         //
         //  TODO: What does this block do, and why?
         //
         if (state) { // We are in our template-id.
            if ((state == npara) && (c != '*')) {
               result = 1; // Tell caller to skip rest of template arguments.
            }
            ++state; // Count the number of times we matched a template parameter since our template-id started.
         }
         break; // Done, we found our match.
      }
      state = 0; // No match, reset count.
      if (tmpl_arg_list) {
         tmpl_arg_list = tmpl_arg_list->next;
      }
   }
   // this is only workaround for STL Allocator
   if (!strcmp(symbol, "Allocator")) {
      strcpy(symbol, G__Allocator);
   }
   return result;
}

//______________________________________________________________________________
//858//static int G__gettemplatearglist(char* tmpl_arg_string, G__Charlist* tmpl_arg_list_in, G__Templatearg* tmpl_para_list, int* pnpara, int parent_tagnum, G__Definedtemplateclass* specialization)
static int G__gettemplatearglist(char* tmpl_arg_string, G__Charlist* tmpl_arg_list_in, G__Templatearg* tmpl_para_list, int* pnpara, int parent_tagnum)
{
   // Convert a template argument string to a list of arguments in canonical form, defaults are included.
   //
   // Returns:
   //
   //   tmpl_para_list: A list of the given template arguments in canonical form.
   //   pnpara: Count of non-default arguments processed.
   //
   if (!tmpl_arg_string) {
      G__genericerror("Error: No template argument list given!");
      return 0;
   }
   //
   //  Temporary buffers.
   //
   G__StrBuf string_sb(G__LONGLINE);
   char* string = string_sb;
   G__StrBuf temp_sb(G__LONGLINE);
   char* temp = temp_sb;
   //
   //
   //  FIXME: Port the following four lines back to cint5?
   //
#if 0 //858//
   G__Templatearg* specArg = 0; // specialized parameters list
   if (specialization) {
      specArg = specialization->spec_arg;
   }
#endif // 0
   //
   //  Loop over provided template argument string and
   //  parse each given argument into canonical form.
   //
   int searchflag = 0; // 1 = arg was modified, 3 = arg was defaulted
   int isrc = 0; // offset into template argument string while parsing
   int c = ',';
   if (!tmpl_arg_string[0] || ((tmpl_arg_string[0] == '>') && !tmpl_arg_string[1])) {
      c = '>';
   }
   G__Charlist* tmpl_arg = tmpl_arg_list_in;
   G__Templatearg* tmpl_param = tmpl_para_list;
   for ( ; c == ','; tmpl_param = tmpl_param->next) { // Loop over all given arguments.
      if (!tmpl_param) { // too many template arguments, error
         G__genericerror("Error: Too many template arguments");
         break;
      }
      c = G__getstream_template(tmpl_arg_string, &isrc, string, ",>\0"); // read next template argument
#if 0 //858//
      // FIXME: Port the following if block to cint5?
      if (specArg && specArg->string && !strcmp(specArg->string, string)) { // Given arg exactly matched a specialized parameter, don't change it.  FIXME: We are skipping the canonicalization step here.
         ++(*pnpara); // Increment the number of arguments processed.
         specArg = specArg->next;
         continue;
      }
#endif // 0
      switch (tmpl_param->type) { // try to rewrite the argument in a canonical way
         case G__TMPLT_CLASSARG: // template <class T>
            strcpy(temp, string); // make a copy of the typename argument
            G__templatemaptypename(temp); // and check if it needs to be changed
            if (strcmp(temp, string)) { // typename changed
               searchflag = 1; // Flag that a template argument was rewritten.
               strcpy(string, temp); // copy over new typename
            }
            break;
         case G__TMPLT_TMPLTARG: // template <template <class U> class T>
            break; // not supported
         case G__TMPLT_POINTERARG3: // template <class T***>
            if (string[0] && (string[strlen(string)-1] == '*')) {
               string[strlen(string)-1] = '\0';
            }
            else {
               G__genericerror("Error: this template requests pointer arg 3");
            }
         case G__TMPLT_POINTERARG2: // template <class T**>
            if (string[0] && (string[strlen(string)-1] == '*')) {
               string[strlen(string)-1] = '\0';
            }
            else {
               G__genericerror("Error: this template requests pointer arg 2");
            }
         case G__TMPLT_POINTERARG1: // template <class T*>
            if (string[0] && (string[strlen(string)-1] == '*')) {
               string[strlen(string)-1] = '\0';
            }
            else {
               G__genericerror("Error: this template requests pointer arg 1");
            }
            break;
         default: // template <int V>
            {
               ::Reflex::Scope store_memberfunc_tagnum = G__memberfunc_tagnum;
               int store_exec_memberfunc = G__exec_memberfunc;
               int store_no_exec_compile = G__no_exec_compile;
               int store_asm_noverflow = G__asm_noverflow;
               G__no_exec_compile = 0;
               G__asm_noverflow = 0;
               if (G__tagdefining && !G__tagdefining.IsTopScope()) {
                  G__exec_memberfunc = 1;
                  G__memberfunc_tagnum = G__tagdefining;
               }
               G__value buf = G__getexpr(string);
               G__no_exec_compile = store_no_exec_compile;
               G__asm_noverflow = store_asm_noverflow;
               G__exec_memberfunc = store_exec_memberfunc;
               G__memberfunc_tagnum = store_memberfunc_tagnum;
               G__string(buf, temp);
               if (strcmp(temp, string)) { // given argument was changed
                  searchflag = 1; // Flag that arg was modified.
                  strcpy(string, temp);
               }
            }
            break;
      }
      //
      //  Output the given argument in canonical form.
      //
      tmpl_arg->string = (char*) malloc(strlen(string) + 1);
      strcpy(tmpl_arg->string, string);
      //
      //  And allocate a new empty entry to mark the end.
      //
      tmpl_arg->next = new G__Charlist;
      tmpl_arg = tmpl_arg->next;
      ++(*pnpara); // Increment the number of arguments processed.
   }
   //
   //  Complete the output argument list with defaults.
   //
   ::Reflex::Scope store_tagdefining = G__tagdefining;
   ::Reflex::Scope store_def_tagnum = G__def_tagnum;
   if (parent_tagnum != -1) {
      G__tagdefining = G__Dict::GetDict().GetScope(parent_tagnum);
      G__def_tagnum = G__Dict::GetDict().GetScope(parent_tagnum);
   }
   for (; tmpl_param; tmpl_param = tmpl_param->next) { // Loop over any remaining template params, which must be defaulted.
      if (!tmpl_param->default_parameter) { // No default available, error.
         G__genericerror("Error: Too few template arguments"); // FIXME: Actually no default available.
         continue;
      }
      strcpy(string, tmpl_param->default_parameter);
      tmpl_arg->string = G__expand_def_template_arg(string, tmpl_para_list, tmpl_arg_list_in);
      {
         // Enlarge the string buffer, G__templatemaptypename may make the typename longer.
         int len = 2 * strlen(tmpl_arg->string);
         if (len < G__LONGLINE) {
            len = G__LONGLINE;
         }
         tmpl_arg->string = (char*) realloc(tmpl_arg->string, len + 1);
         G__templatemaptypename(tmpl_arg->string);
         G__ASSERT((int) strlen(tmpl_arg->string) <= (int) len);
      }
      tmpl_arg->next = new G__Charlist;
      tmpl_arg = tmpl_arg->next;
      searchflag = 3; // Flag that a template arg was defaulted.
   }
   G__tagdefining = store_tagdefining;
   G__def_tagnum = store_def_tagnum;
   return searchflag;
}

//______________________________________________________________________________
static void G__templatemaptypename(char* string)
{
   // Normalize a passed typename string.  Intended for processing template arguments.
   if (!strncmp(string, "const", 5) && (string[5] != ' ')) {
      if (
         !strcmp(string + 5, "int") ||
         !strcmp(string + 5, "unsignedint") ||
         !strcmp(string + 5, "char") ||
         !strcmp(string + 5, "unsignedchar") ||
         !strcmp(string + 5, "short") ||
         !strcmp(string + 5, "unsignedshort") ||
         !strcmp(string + 5, "long") ||
         !strcmp(string + 5, "unsignedlong") ||
         !strcmp(string + 5, "double") ||
         !strcmp(string + 5, "float") ||
         !strcmp(string + 5, "int*") ||
         !strcmp(string + 5, "unsignedint*") ||
         !strcmp(string + 5, "char*") ||
         !strcmp(string + 5, "unsignedchar*") ||
         !strcmp(string + 5, "short*") ||
         !strcmp(string + 5, "unsignedshort*") ||
         !strcmp(string + 5, "long*") ||
         !strcmp(string + 5, "unsignedlong*") ||
         !strcmp(string + 5, "double*") ||
         !strcmp(string + 5, "float*") ||
         G__istypename(string + 5)
      ) {
         int len = strlen(string);
         while (len >= 5) {
            string[len+1] = string[len];
            --len;
         }
         string[5] = ' ';
         string += 6;
      }
   }
   while (!strncmp(string, "const ", 6)) {
      string += 6;
   }
   if (!strcmp(string, "shortint")) {
      strcpy(string, "short");
   }
   else if (!strcmp(string, "shortint*")) {
      strcpy(string, "short*");
   }
   else if (!strcmp(string, "longint")) {
      strcpy(string, "long");
   }
   else if (!strcmp(string, "longint*")) {
      strcpy(string, "long*");
   }
   else if (!strcmp(string, "longlong")) {
      strcpy(string, "long long");
   }
   else if (!strcmp(string, "longlong*")) {
      strcpy(string, "long long*");
   }
   else if (!strcmp(string, "unsignedchar")) {
      strcpy(string, "unsigned char");
   }
   else if (!strcmp(string, "unsignedchar*")) {
      strcpy(string, "unsigned char*");
   }
   else if (!strcmp(string, "unsignedint")) {
      strcpy(string, "unsigned int");
   }
   else if (!strcmp(string, "unsignedint*")) {
      strcpy(string, "unsigned int*");
   }
   else if (!strcmp(string, "unsignedlong") || !strcmp(string, "unsignedlongint")) {
      strcpy(string, "unsigned long");
   }
   else if (!strcmp(string, "unsignedlong*") || !strcmp(string, "unsignedlongint*")) {
      strcpy(string, "unsigned long*");
   }
   else if (!strcmp(string, "unsignedlonglong")) {
      strcpy(string, "unsigned long long ");
   }
   else if (!strcmp(string, "unsignedlonglong*")) {
      strcpy(string, "unsigned long long*");
   }
   else if (!strcmp(string, "unsignedshort") || !strcmp(string, "unsignedshortint")) {
      strcpy(string, "unsigned short");
   }
   else if (!strcmp(string, "unsignedshort*") || !strcmp(string, "unsignedshortint*")) {
      strcpy(string, "unsigned short*");
   }
   else if (!strcmp(string, "Float16_t") || !strcmp(string, "Float16_t*")) {
      // nothing to do, we want to keep those as is
   }
   else if (!strcmp(string, "Double32_t") || !strcmp(string, "Double32_t*")) {
      // nothing to do, we want to keep those as is
   }
   else {
      G__StrBuf saveref_sb(G__LONGLINE);
      char* saveref = saveref_sb;
      char* p = string + strlen(string);
      while ((p > string) && ((p[-1] == '*') || (p[-1] == '&'))) {
         --p;
      }
      G__ASSERT(strlen(p) < sizeof(saveref));
      strcpy(saveref, p);
      *p = '\0';
      ::Reflex::Type typenum = G__find_typedef(string);
      if (typenum) {
         char type = G__get_type(typenum);
         int ref = G__get_reftype(typenum);
         if (!strstr(string, "::") && (typenum.DeclaringScope() != ::Reflex::Scope::GlobalScope())) { // The arg was unqualified and its type has a parent, flag this.
            // The arg was unqualified and its type has
            // a parent, flag this for G__instantiate_templateclass
            // so that when it creates the typedef to map the given
            // template name to this changed name, it will create the
            // typedef in the parent of the argument type instead of
            // the parent of the template definition.
            ++G__templatearg_enclosedscope;
         }
         int target_tagnum = G__get_tagnum(typenum);
         if ((target_tagnum != -1) && (G__struct.name[target_tagnum][0] == '$')) {
            type = tolower(type);
         }
         strcpy(string, G__type2string(type, target_tagnum, -1, ref, 0));
      }
      else {
         int tagnum = G__defined_tagname(string, 1);
         if (tagnum != -1) { // Template argument is a defined class, enum, namespace, struct, or union type.
            if (!strstr(string, "::") && (G__struct.parent_tagnum[tagnum] != -1)) { // The arg was unqualified and its type has a parent, flag this.
               // The arg was unqualified and its type has
               // a parent, flag this for G__instantiate_templateclass
               // so that when it creates the typedef to map the given
               // template name to this changed name, it will create the
               // typedef in the parent of the argument type instead of
               // the parent of the template definition.
               ++G__templatearg_enclosedscope;
            }
            strcpy(string, G__fulltagname(tagnum, 1));
         }
      }
      strcat(string, saveref);
   }
}

//______________________________________________________________________________
static char* G__expand_def_template_arg(char* str_in, G__Templatearg* def_para, G__Charlist* charlist)
{
   // Returns a malloc'd string.
   const char* punctuation = " \t\n;:=+-)(*&^%$#@!~'\"\\|][}{/?.>,<";
   int siz_out = strlen(str_in) * 2;
   char* str_out;
   char* temp;
   int iout;
   int iin;
   int single_quote;
   int double_quote;
   char c;
   int isconst = 0;
   if (siz_out < 10) {
      siz_out = 10;
   }
   temp = (char*) malloc(siz_out + 1);
   str_out = (char*) malloc(siz_out + 1);
   str_out[0] = 0;
   iout = 0;
   iin = 0;
   // The text has been through the reader once, so we shouldn't
   // have to worry about comments.
   // We should still be prepared to handle quotes though.
   single_quote = 0;
   double_quote = 0;
   do {
      int lreslt;
      char* reslt = temp;
      c = G__getstream(str_in, &iin, temp, punctuation);
      if (*reslt && !single_quote && !double_quote) {
         G__Charlist* cl = charlist;
         G__Templatearg* ta = def_para;
         while (cl && cl->string) {
            G__ASSERT(ta && ta->string);
            if (!strcmp(ta->string, reslt)) {
               reslt = cl->string;
               break;
            }
            ta = ta->next;
            cl = cl->next;
         }
      }
      // ??? Does this handle backslash escapes properly?
      if ((c == '\'') && !double_quote) {
         single_quote = single_quote ^ 1;
      }
      else if ((c == '"') && !single_quote) {
         double_quote = double_quote ^ 1;
      }
      lreslt = strlen(reslt);
      if ((iout + lreslt + 1) > siz_out) {
         siz_out = (iout + lreslt + 1) * 2;
         str_out = (char*) realloc(str_out, siz_out + 1);
      }
      {
         int rlen = strlen(reslt);
         if (isconst &&
             !strncmp(reslt, "const ", 6) &&
             (rlen > 0) &&
             (reslt[rlen-1] == '*')) {
            strcpy(str_out + iout, reslt + 6);
            strcat(str_out, " const");
            iout += lreslt;
            isconst = 0;
         }
         else if (
            isconst &&
            (iout >= 6) &&
            !strncmp(str_out + iout - 6, "const ", 6) &&
            (rlen > 0) &&
            (reslt[rlen-1] == '*')
         ) {
            strcpy(str_out + iout - 6, reslt);
            strcat(str_out, " const");
            iout += lreslt;
            isconst = 0;
         }
         else if (
            isconst &&
            (iout >= 6) &&
            !strncmp(str_out + iout - 6, "const ", 6) &&
            !strncmp(reslt, "const ", 6)
         ) {
            // const T with T="const A" becomes "const const A", so skip one const
            strcpy(str_out + iout, reslt + 6);
            iout += lreslt - 6;
            isconst = 0;
         }
         else {
            strcpy(str_out + iout, reslt);
            iout += lreslt;
            isconst = 0;
            if (!strcmp(reslt, "const") && (c == ' ')) {
               isconst = 1;
            }
         }
      }
      str_out[iout++] = c;
   }
   while (c != '\0');
   str_out[iout] = '\0';
   free(temp);
   return str_out;
}

//______________________________________________________________________________
static G__Templatearg* G__read_specializationarg(G__Templatearg* tmpl_params, char* source)
{
   // template<class T,class E,int S> ...
   //          ^
   G__Templatearg* targ = 0;
   G__Templatearg* p = 0;
   G__StrBuf type_sb(G__MAXNAME);
   char* type = type_sb;
   bool done = false;
   int isrc = 0;
   int len;
   do {
      // allocate entry of template argument list
      if (!p) {
         p = new G__Templatearg;
         p->next = 0;
         p->type = 0;
         p->string = 0;
         p->default_parameter = 0;
         targ = p;
      }
      else {
         p->next = new G__Templatearg;
         p = p->next;
         p->type = 0;
         p->string = 0;
         p->default_parameter = 0;
         p->next = 0;
      }
      //  templatename<T*,E,int> ...
      //                ^
      if (!strncmp(source + isrc, "const ", 6)) {
         p->type |= G__TMPLT_CONSTARG;
         isrc += 6;
      }
      len = strlen(source);
      unsigned int newlen = 0;
      { // additional block against MSVC 7.1 warning "conflicting var decl for i in for loop and in line 1720"
         for (int i = isrc, nest = 0; i < len; ++i) {
            switch (source[i]) {
            case '<':
               ++nest;
               break;
            case '>':
               --nest;
               if (nest < 0) {
                  i = len;
                  done = true;
                  continue;
               }
               break;
            case ',':
               if (!nest) {
                  isrc = i + 1;
                  i = len;
                  continue;
               }
               break;
            }
            type[newlen++] = source[i];
         }
      }
      type[newlen] = 0;
      len = newlen;
      //
      //  Accumulate and remove a possible reference qualifier.
      //
      if (type[len-1] == '&') {
         p->type |= G__TMPLT_REFERENCEARG;
         type[--len] = 0;
      }
      //
      //  Accumulate and remove any pointer qualifiers.
      //
      while (type[len-1] == '*') {
         p->type += G__TMPLT_POINTERARG1;
         type[--len] = 0;
      }
      //
      //  Exit if no arguments.
      //
      if (!strcmp(type, ">")) {
         delete targ;
         return 0;
      }
      //
      //  FIXME: long long and unsigned long long are missing!
      //
      //  TODO: template template arguments are not handled yet.
      //
      if (!strcmp(type, "char")) {
         p->type |= G__TMPLT_CHARARG;
      }
      else if (!strcmp(type, "unsigned char")) {
         p->type |= G__TMPLT_UCHARARG;
      }
      else if (!strcmp(type, "short")) {
         p->type |= G__TMPLT_SHORTARG;
      }
      else if (!strcmp(type, "unsigned short")) {
         p->type |= G__TMPLT_USHORTARG;
      }
      else if (!strcmp(type, "int")) {
         p->type |= G__TMPLT_INTARG;
      }
      else if (!strcmp(type, "unsigned int")) {
         p->type |= G__TMPLT_UINTARG;
      }
      else if (!strcmp(type, "long")) {
         p->type |= G__TMPLT_LONGARG;
      }
      else if (!strcmp(type, "unsigned long")) {
         p->type |= G__TMPLT_ULONGARG;
      }
      else if (!strcmp(type, "unsigned")) {
         p->type |= G__TMPLT_UINTARG;
      }
      else if (!strcmp(type, "size_t")) { // FIXME: Cint extension!
         p->type |= G__TMPLT_SIZEARG;
      }
      else if (!strcmp(type, "float")) { // FIXME: Cint extension!
         p->type |= G__TMPLT_FLOATARG;
      }
      else if (!strcmp(type, "double")) { // FIXME: Cint extension!
         p->type |= G__TMPLT_DOUBLEARG;
      }
      else {
         p->type |= G__TMPLT_CLASSARG; // TODO: Could be a template template argument instead of a type template argument here!
      }
      //
      //  Check first if this is a reference to
      //  a specialization parameter by name.
      //
      bool found = false;
      int i = 1;
      for (G__Templatearg* param = tmpl_params; param; param = param->next, ++i) {
         if (strstr(type, param->string)) { // FIXME: Terrible, terrible hack.  Checking for a type name that is dependent on the specialization parameters using a substring match, way too broad, lots of false positives.
            found = true; // Make sure we do not attempt to canonicalize the argument.
            if (!strcmp(type, param->string)) { // The specialization argument is a reference to a specialization parameter by name.
               G__StrBuf temp_sb(G__LONGLINE);
               char* temp = temp_sb;
               sprintf(temp, "%s,//P%d//", type, i);
               p->string = (char*) malloc(strlen(temp) + 1);
               strcpy(p->string, temp);
               break;
            }
            else { // Not an exact match, just copy it.  It will not be found at instantiation time and the specialization instantiation attempt will fail.  We cannot handle it yet.
               p->string = (char*) malloc(strlen(type) + 1);
               strcpy(p->string, type);
               break;
            }
         }
      }
      if (!found) {// Not a reference to a specialization parameter by name.
         if ( // Rewrite the type name in canonical form.
            ((p->type & 0xff) == G__TMPLT_CLASSARG) ||
            ((p->type & 0xff) == G__TMPLT_TMPLTARG)
         ) { // Rewrite the type name in canonical form.
            if (!found) { // Not a reference to a specialization template parameter.
               G__StrBuf temp_sb(G__LONGLINE);
               char* temp = temp_sb;
               strcpy(temp, type);
               G__templatemaptypename(temp); // Rewrite the type name in canonical form.
               p->string = (char*) malloc(strlen(temp) + 1);
               strcpy(p->string, temp);
            }
         }
         else { // A non-type specialization argument, type name is already canonical.
            p->string = (char*) malloc(strlen(type) + 1);
            strcpy(p->string, type);
         }
      }
      //  template<T*,E,int> ...
      //              ^
   }
   while (!done);
   //  template<T*,E,int> ...
   //                   ^
   return targ;
}

//______________________________________________________________________________
static int G__checkset_charlist(char* type_name, G__Charlist* pcall_para, int narg, int ftype)
{
   // Check and set actual template argument
   for (int i = 1; i < narg; ++i) {
      if (!pcall_para->next) {
         pcall_para->next = new G__Charlist;
      }
      pcall_para = pcall_para->next;
   }
   if (pcall_para->string) {
      if (ftype == 'U') {
         int len = strlen(type_name);
         if (len && (type_name[len-1] == '*')) {
            type_name[len-1] = '\0';
            if (!strcmp(type_name, pcall_para->string)) {
               type_name[len-1] = '*';
               return 1;
            }
            type_name[len-1] = '*';
         }
      }
      if (!strcmp(type_name, pcall_para->string)) {
         return 1;
      }
      return 0;
   }
   pcall_para->string = (char*) malloc(strlen(type_name) + 1);
   strcpy(pcall_para->string, type_name);
   if (ftype == 'U') {
      int len = strlen(type_name);
      if (len && (type_name[len-1] == '*')) {
         pcall_para->string[len-1] = '\0';
      }
   }
   return 1;
}

//______________________________________________________________________________
static int G__matchtemplatefunc(G__Definetemplatefunc* deftmpfunc, G__param* libp, G__Charlist* pcall_para, int funcmatch)
{
   // Test if given function arguments and template function arguments matches.
   int fparan;
   int paran;
   int ftype;
   int type;
   int ftagnum;
   int tagnum;
   ::Reflex::Type ftypenum;
   ::Reflex::Type typenum;
   int freftype;
   int reftype;
   int ref;
   int fargtmplt;
   G__StrBuf paratype_sb(G__LONGLINE);
   char* paratype = paratype_sb;
   int* fntarg;
   int fnt;
   char** fntargc;
   fparan = deftmpfunc->func_para.paran;
   paran = libp->paran;
   // more argument in calling function, unmatch
   if (paran > fparan) {
      return 0;
   }
   if (fparan > paran) {
      if (!deftmpfunc->func_para.paradefault[paran]) {
         return 0;
      }
   }
   for (int i = 0; i < paran; ++i) {
      // get template information for simplicity
      ftype = deftmpfunc->func_para.type[i];
      ftagnum = deftmpfunc->func_para.tagnum[i];
      ftypenum = deftmpfunc->func_para.typenum[i];
      freftype = deftmpfunc->func_para.reftype[i];
      fargtmplt = deftmpfunc->func_para.argtmplt[i];
      fntarg = deftmpfunc->func_para.ntarg[i];
      fnt = deftmpfunc->func_para.nt[i];
      fntargc = deftmpfunc->func_para.ntargc[i];
      // get parameter information for simplicity
      ::Reflex::Type valuetype(G__value_typenum(libp->para[i]).FinalType());
      type = G__get_type(valuetype); // Or is tagtype
      tagnum = G__get_tagnum(valuetype);
      typenum = valuetype;
      ref = libp->para[i].ref;
      if ((type == 'u') || valuetype.IsPointer()) {
         reftype = G__get_reftype(G__value_typenum(libp->para[i]));
      }
      else {
         reftype = G__PARANORMAL;
      }
      // match parameter
      if (fargtmplt == -1) {
         char* p;
         char* cntarg[20];
         int cnt = 0;
         int j;
         int basetagnum;
         int basen;
         int bn;
         int bmatch;
         // fixed argument type
         if ((type == ftype) && (ftagnum == tagnum) && (!freftype || ref || (freftype == reftype))) {
            continue;
         }
         // assuming that the argument type is a template class
         if ((type != 'u') || (tagnum == -1)) {
            return 0;
         }
         // template argument  (T<E> a)
         basen = G__struct.baseclass[tagnum]->vec.size();
         bn = -1;
         basetagnum = tagnum;
         bmatch = 0;
         while (!bmatch && (bn < basen)) {
            int nest = 0;
            cnt = 0;
            if (bn >= 0) {
               basetagnum = G__struct.baseclass[tagnum]->vec[bn].basetagnum;
            }
            ++bn;
            bmatch = 1;
            strcpy(paratype, G__fulltagname(basetagnum, 0));
            cntarg[cnt++] = paratype;  // T <x,E,y>
            p = strchr(paratype, '<');
            if (!p) { // unmatch
               if (funcmatch == G__EXACT) {
                  return 0;
               }
               bmatch = 0;
               continue;
            }
            do { // T<x,E,y>
               *p = 0;  //   ^ ^ ^
               ++p;     //    ^ ^ ^
               cntarg[cnt++] = p;
               while ((*p && (*p != ',') && (*p != '>')) || nest) {
                  if (*p == '<') {
                     ++nest;
                  }
                  else if (*p == '>') {
                     --nest;
                  }
                  ++p;
               }
            }
            while (*p == ',');
            if (*p == '>') {
               *p = 0;  // the last '>'
            }
            if (*(p - 1) == ' ') {
               *(p - 1) = 0;
            }
            // match template argument
            if (fnt > cnt) { // unmatch
               if (funcmatch == G__EXACT) {
                  return 0;
               }
               bmatch = 0;
               continue;
            }
            else if (fnt < cnt) { // unmatch, check default template argument
               int ix;
               G__Templatearg* tmparg;
               G__Definedtemplateclass* tmpcls;
               tmpcls = G__defined_templateclass(paratype);
               if (!tmpcls) {
                  if (funcmatch == G__EXACT) {
                     return 0;
                  }
                  bmatch = 0;
                  continue;
               }
               tmparg = tmpcls->def_para;
               for (ix = 0; (ix < (fnt - 1)) && tmparg; ++ix) {
                  tmparg = tmparg->next;
               }
               // Note: This one is a correct behavior. Current implementation is
               // workaround for old and new STL mixture
               //  if(!tmparg || !tmparg->default_parameter) {}
               if (tmparg && !tmparg->default_parameter) {
                  if (funcmatch == G__EXACT) {
                     return 0;
                  }
                  bmatch = 0;
                  continue;
               }
            }
            for (j = 0; (j < fnt) && (j < cnt); ++j) {
               if (fntarg[j]) {
                  if (G__checkset_charlist(cntarg[j], pcall_para, fntarg[j], ftype)) {
                     // match or newly set template argument
                  }
                  else {
                     // template argument is already set to different type, unmatch
                     if (funcmatch == G__EXACT) {
                        return 0;
                     }
                     bmatch = 0;
                     break;
                  }
               }
               else if (!fntargc[j] || strcmp(cntarg[j], fntargc[j])) {
                  if (funcmatch == G__EXACT) {
                     return 0;
                  }
                  bmatch = 0;
                  break;
               }
            }
         }
         if (!bmatch) {
            return 0;
         }
      }
      else if (fargtmplt) {
         if (isupper(ftype) && islower(type)) {
            // unmatch , pointer level f(T* x) <= f(1)
            return 0;
         }
         // template argument  (T a)
         if (reftype == G__PARAREFERENCE) {
            strcpy(paratype, G__type2string(type, tagnum, -1, 0, 0));
         }
         else {
            strcpy(paratype, G__type2string(type, tagnum, -1, reftype, 0));
         }
         if (!strncmp(paratype, "class ", 6)) {
            int subi = 6;
            int subj = 0;
            do {
               paratype[subj++] = paratype[subi];
            }
            while (paratype[subi++]);
         }
         else if (!strncmp(paratype, "struct ", 7)) {
            int subi = 7;
            int subj = 0;
            do {
               paratype[subj++] = paratype[subi];
            }
            while (paratype[subi++]);
         }
         if (G__checkset_charlist(paratype, pcall_para, fargtmplt, ftype)) {
            // match or newly set template argument
         }
         else {
            // template argument is already set to different type, unmatch
            return 0;
         }
      }
      else {
         // fixed argument type
         if ((type == ftype) && (ftagnum == tagnum) && (!freftype || ref)) {
            // match, check next
         }
         else if (
            (G__EXACT != funcmatch) &&
            (
               (
                  (type == 'u') &&
                  (ftype == 'u')
               ) ||
               (
                  (type == 'U') &&
                  (ftype == 'U')
               )
            ) &&
            (G__ispublicbase(tagnum, ftagnum, (void*)libp->para[i].obj.i) != -1)
         ) {
            // match with conversion
         }
         else {
            // unmatch
            return 0;
         }
      }
   }
   return 1; // All parameters match
}

//______________________________________________________________________________
//______________________________________________________________________________
//
//  Parsing
//

//______________________________________________________________________________
void Cint::Internal::G__declare_template()
{
   //  Parse a template declaration.
   //
   //  template<class T> class A { };
   //           ^
   //  template<class T> type A <T>::f() { }
   //  template<class T> A <T>::B<T> A <T>::f() { }
   //  template<class T> A <T>::A() { }
   //           ^
   //  template<class T> type A <T>::staticmember;
   //  template<class T> A <T>::B<T> A <T>::staticmember;
   //           ^
   //  template<class T> type f() { }
   //  template<class T> A <T>::B<T> f() { }
   //           ^
   //
   G__StrBuf temp_sb(G__LONGLINE);
   char* temp = temp_sb;
   fpos_t pos;
   int store_line_number;
   G__Templatearg *targ;
   int c;
   char *p;
   G__StrBuf temp2_sb(G__LONGLINE);
   char* temp2 = temp2_sb;
   G__StrBuf temp3_sb(G__LONGLINE);
   char* temp3  = temp3_sb;
   int ismemvar = 0;
   int isforwarddecl = 0;
   int isfrienddecl = 0;
   int autoload_old = 0;
   //
   //
   //
   if (G__ifile.filenum > G__gettempfilenum()) {
      G__fprinterr(G__serr, "Limitation: template can not be defined in a command line or a tempfile\n");
      G__genericerror("You need to write it in a source file");
      return;
   }
   //
   //   Set a flag that template or macro is included in the source file,
   //   so that this file won't be closed even with -cN option after preRUN
   //
   ++G__macroORtemplateINfile;
   //
   //  Read template parameters (formal arguments);
   //
   targ = G__read_formal_templatearg();
   //
   //  Handle any explicit specialization request.
   //
   if (!targ) { // in case of 'template<>'
      G__explicit_template_specialization();
      return;
   }
   //
   // Parsing state is now:
   //      template<class T,class E,int S> ...
   //                                     ^
   //
   // Remember this file position.
   fgetpos(G__ifile.fp, &pos);
   store_line_number = G__ifile.line_number;
   //
   //  Skip past any "inline", "const", or "typename"
   //  keywords while handling "friend".
   //
   //  FIXME: We do not handle function templates with const return values correctly here.  Or do we, does a const return value really mean anything?
   //
   do {
      c = G__fgetname_template(temp, "(<");
      if (!strcmp(temp, "friend")) { // Handle friend template declarations;
         isfrienddecl = 1;
         autoload_old = G__set_class_autoloading(0); // Flag that we do not need to autoload friend template declarations.
         c = G__fgetname_template(temp, "(<");
      }
   }
   while (!strcmp(temp, "inline") || !strcmp(temp, "const") || !strcmp(temp, "typename"));
   //
   //  Handle a class template declaration or definition.
   //
   if (!strcmp(temp, "class") || !strcmp(temp, "struct")) { // FIXME: We need to support "union" too!
      // Class or struct template.
      fpos_t fppreclassname;
      fgetpos(G__ifile.fp, &fppreclassname);
      c = G__fgetstream_template(temp, ":{;"); // read template name and any possible specialization arguments
      bool haveFuncReturn = false; // whether we have "class A<T>::B f()"
      if (c == ';') { // Nothing after the name, this is a forward declaration.
         isforwarddecl = 1;
         if (isfrienddecl) { // Friend declarations are NOT forward declarations, return.
            G__set_class_autoloading(autoload_old); // Restore the old autoloading setting.
            return;
         }
      } else if (c == ':') {
         fpos_t fpprepeek;
         fgetpos(G__ifile.fp, &fpprepeek);
         // could be "class A<T>::B f()" i.e. not a template class but
         // a function with a templated return type.
         char c2 = G__fgetc();
         if (c2 == ':') {
            haveFuncReturn = true;
            // put temp back onto the stream, get up to '<'
            fsetpos(G__ifile.fp, &fppreclassname);
            c = G__fgetname_template(temp,"(<");
         } else
            fsetpos(G__ifile.fp, &fpprepeek);
      }
      if (!haveFuncReturn) {
         //
         //  Rewind file position to just after
         //  the template arguments.
         //
         fsetpos(G__ifile.fp, &pos);
         if (G__dispsource) {
            G__disp_mask = 0;
         }
         G__ifile.line_number = store_line_number;
         //
         //  Make an entry in the class template table.
         //
         G__createtemplateclass(temp, targ, isforwarddecl);
         //
         //  Restore any changes to autoloading and we are done.
         //
         if (isfrienddecl) { // This was a friend class template definition.
            G__set_class_autoloading(autoload_old); // Reset old autoloading flag.
         }
         return;
      }
   }
   //
   //  Not a class template declaration or definition, so this
   //  must be a function template declaration or definition.
   //
   //  Now, determine whether or not this is a member function template.
   //
   if (c == '<') {
      // must judge if this is a constructor or other function
      //1 template<class T> A<T>::f()  constructor
      //2 template<class T> A<T>::B<T> A<T>::f()
      //3 template<class T> A<T> A<T>::f()
      //4 template<class T> A<T>::B<T> f()
      //5 template<class T> A<T> f()
      //6 template<class T> A<T> A<T>::v;
      //6'template<class T> A<T> A<T>::v = 0;
      //7 template<class T> A<T> { }  constructor
      //  also the return value could be a pointer or reference or const
      //  or any combination of the 3
      //                      ^>^
      c = G__fgetstream_template(temp3, ">");
      c = G__fgetname_template(temp2, "*&(;");
      if ((c == '*') && !strncmp(temp2, "operator", strlen("operator"))) {
         strcat(temp2, "*");
         c = G__fgetname_template(temp2 + strlen(temp2), "*&(;=");
      }
      else if ((c == '&') && !strncmp(temp2, "operator", strlen("operator"))) {
         strcat(temp2, "&");
         c = G__fgetname_template(temp2 + strlen(temp2), "*(;=");
      }
      while ((c == '&') || (c == '*')) {
         // we skip all the & and * we see and what's in between.
         // This should be removed from the func name (what we are looking for)
         // anything preceding combinations of *,& and const.
         c = G__fgetname_template(temp2, "*&(;=");

         size_t len = strlen(temp2);
         static size_t oplen( strlen( "::operator" ) ); 
         if ((  !strncmp(temp2,"operator",strlen("operator"))
              ||(len>=oplen && !strncmp(temp2+(len-oplen),"::operator",oplen)))
             && strchr("&*=", c)) {
            while ((c == '&') || (c == '*') || (c == '=')) {
               temp2[len + 1] = 0;
               temp2[len] = c;
               ++len;
               c = G__fgetname_template(temp2 + len, "*&(;=");
            }
         }
      }
      if (!temp2[0]) { // constructor template in class definition
         strcat(temp, "<");
         strcat(temp, temp3);
         strcat(temp, ">");
      }
      if (isspace(c)) {
         if (!strcmp(temp2, "::~")) {
            c = G__fgetname_template(temp2 + 3, "(;");
         }
         else if (!strcmp(temp2, "::")) {
            c = G__fgetname_template(temp2 + 2, "(;");
         }
         else if ((p = strstr(temp2, "::")) && !strcmp(p, "::operator")) {
            // A<T> A<T>::operator T () { }
            c = '<'; // this is a flag indicating this is a member function tmplt
         }
         else if (!strcmp(temp2, "operator")) {
            c = G__fgetstream(temp2 + 8, "(");
         }
      }
      if ((c == ';') || (c == '=')) {
         ismemvar = 1;
      }
      if ((c == '(') || (c == ';') || (c == '=')) {
         //1 template<class T> A<T>::f()           ::f
         //3 template<class T> A<T> A<T>::f()      A<T>::f
         //6 template<class T> A<T> A<T>::v;       A<T>::v
         //6'template<class T> A<T> A<T>::v=0;     A<T>::v
         //7 template<class T> A<T> { }  constructor
         //5 template<class T> A<T> f()            f
         p = strchr(temp2, ':');
         if (p) {
            c = '<';
            if (p != temp2) {
               p = strchr(temp2, '<');
               *p = '\0'; // non constructor/destructor member function
               strcpy(temp, temp2);
            }
         }
         else {
            if (temp2[0]) {
               strcpy(temp, temp2);
            }
         }
      }
      else if (c == '<') {
         // Do nothing
      }
      else {
         //2 template<class T> A<T>::B<T> A<T>::f()  ::B<T>
         //4 template<class T> A<T>::B<T> f()        ::B<T>
         // take out keywords const
         fpos_t posx;
         int linex;
         G__disp_mask = 1000;
         fgetpos(G__ifile.fp, &posx);
         linex = G__ifile.line_number;
         c = G__fgetname(temp, "&*(;<");
         if (!strcmp(temp, "const")) {
            G__constvar = G__CONSTVAR;
            if (G__dispsource) {
               G__fprinterr(G__serr, "%s", temp);
            }
            if (!isspace(c)) {
               fseek(G__ifile.fp, -1, SEEK_CUR);
            }
         }
         else {
            G__disp_mask = 0;
            fsetpos(G__ifile.fp, &posx);
            G__ifile.line_number = linex;
         }
         c = G__fgetstream(temp, "(;<");
         // Judge by c? '('  global or '<' member
      }
   }
   // template<...> X() in class context could be a ctor.
   // template<...> X::X() outside class handled below
   else if (
      (c == '(') &&
      G__def_struct_member &&
      (G__tagdefining && !G__tagdefining.IsTopScope()) &&
      (G__tagdefining.Name() == temp)
   ) {
      //8 template<class T> A(const T& x) { }  constructor
      //                      ^
      // Do nothing
   }
   else if (isspace(c) && !strcmp(temp, "operator")) {
      unsigned int len = 8;
      do {
         temp[len++] = ' ';
         temp[len] = '\0';
         
         char* ptr = temp + len;
         c=G__fgetname_template(ptr,"(");
         len = strlen(temp);
         if (len >= G__LONGLINE)
            {
               temp[G__LONGLINE-1] = '\0';
               G__fprinterr(G__serr,"line too long. '%s'\n", temp);
               break;
            }
      } while (c != '(');
   }
   else if ((c == '(') && strstr(temp, "::")) {
      // template<..> inline A::A(T a,S b) { ... }
      //                          ^
      std::string classname(temp);
      size_t posLastScope = std::string::npos;
      for (
         size_t posScope = classname.find("::");
         posScope != std::string::npos;
         posScope = classname.find("::", posScope + 2)
      ) {
         posLastScope = posScope;
      }
      std::string funcname(classname.substr(posLastScope + 2));
      if (classname.compare(posLastScope - funcname.length(), funcname.length(), funcname)) {
         G__fprinterr(G__serr, "Error: expected templated constructor, got a templated function with a return type containing a '(': %s\n", temp);
         // try to ignore it...
      }
      else {
         // do nothing, just like for the in-class case.
      } // c'tor?
   }
   else {
      // template<..> inline|const type A<T,S>::f() { ... }
      // template<..> inline|const type  f(T a,S b) { ... }
      //                               ^
      do {
         c = G__fgetname_template(temp, "(<&*");
         if (!strcmp(temp, "operator")) {
            if (isspace(c)) {
               c = G__fgetstream(temp + 8, "(");
               if ((c == '(') && !strcmp(temp, "operator(")) {
                  c = G__fgetname(temp + 9, "(");
               }
            }
            else if ((c == '&') || (c == '*')) {
               temp[8] = c;
               temp[9] = 0;
               c = G__fgetstream(temp + 9, "(");
            }
         }
      }
      while ((c != '(') && (c != '<'));
   }
   // template<..> type A<T,S>::f() { ... }
   // template<..> type f(T a,S b) { ... }
   //                     ^
   if ((c == '<') && strcmp(temp, "operator")) {
      // member function template
      fsetpos(G__ifile.fp, &pos);
      G__ifile.line_number = store_line_number;
      if (G__dispsource) {
         G__disp_mask = 0;
      }
      G__createtemplatememfunc(temp);
      // skip body of member function template
      c = G__fignorestream("{;");
      if (c != ';') {
         c = G__fignorestream("}");
      }
      G__freetemplatearg(targ);
   }
   else {
      if (G__dispsource) {
         G__disp_mask = 0;
      }
      // global function template
      if (!strcmp(temp, "operator")) {
         // in case of operator< operator<= operator<<
         temp[8] = c; // operator<
         c = G__fgetstream(temp + 9, "(");
         if (temp[8] == '(') {
            if (c == ')') {
               temp[9] = c;
               c = G__fgetstream(temp + 10, "(");
            }
            else {
               G__genericerror("Error: operator() overloading syntax error");
               if (isfrienddecl) {
                  G__set_class_autoloading(autoload_old);
               }
               return;
            }
         }
      }
      G__createtemplatefunc(temp, targ, store_line_number, &pos);
   }
   if (isfrienddecl) {
      G__set_class_autoloading(autoload_old);
   }
}

//______________________________________________________________________________
static G__Templatearg* G__read_formal_templatearg()
{
   // Read and parse a template parameter list.
   //
   // Parse state on entry is:
   //
   //      template <class T,class E,int S> ...
   //                ^
   //--
   G__Templatearg* tmpl_arg_list = new G__Templatearg; // This is our return value.
   tmpl_arg_list->type = 0;
   tmpl_arg_list->string = 0;
   tmpl_arg_list->default_parameter = 0;
   tmpl_arg_list->next = 0;
   G__StrBuf type_sb(G__MAXNAME);
   char* type = type_sb;
   G__StrBuf name_sb(G__MAXNAME);
   char* name = name_sb;
   G__Templatearg* p = tmpl_arg_list;
   bool first = true;
   int c = ',';
   while (c == ',') {
      // Allocate next entry in the template argument list.
      if (first) {
         first = false;
      }
      else {
         p->next = new G__Templatearg;
         p = p->next;
         p->type = 0;
         p->string = 0;
         p->default_parameter = 0;
         p->next = 0;
      }
      //  template<class T,class E,int S> ...
      //           ^
      c = G__fgetname(type, "<");
      if (!strcmp(type, ">")) { // All done, argument list is empty (explicit specialization case).
         delete tmpl_arg_list;
         return 0; // Flag this is an explicit specialization.
      }
      if (!strcmp(type, "const") && (c == ' ')) { // Ignore const in a template parameter.  FIXME: We must ignore volatile as well!
         c = G__fgetname(type, "<");
      }
      //
      //  FIXME: unsigned long long is missing from this list!
      //
      if (!strcmp(type, "class") || !strcmp(type, "typename")) { // This is a type template parameter.
         p->type = G__TMPLT_CLASSARG;
      }
      else if ((c == '<') && !strcmp(type, "template")) { // This is a template template parameter.
         c = G__fignorestream(">");
         c = G__fgetname(type, "");
         G__ASSERT(!strcmp(type, "class") || !strcmp(type, "typename"));
         p->type = G__TMPLT_TMPLTARG;
      }
      else if (!strcmp(type, "char")) {
         p->type = G__TMPLT_CHARARG;
      }
      else if (!strcmp(type, "unsignedchar")) {
         p->type = G__TMPLT_UCHARARG;
      }
      else if (!strcmp(type, "short")) {
         p->type = G__TMPLT_SHORTARG;
      }
      else if (!strcmp(type, "unsignedshort")) {
         p->type = G__TMPLT_USHORTARG;
      }
      else if (!strcmp(type, "int")) {
         p->type = G__TMPLT_INTARG;
      }
      else if (!strcmp(type, "unsignedint")) {
         p->type = G__TMPLT_UINTARG;
      }
      else if (!strcmp(type, "long")) {
         p->type = G__TMPLT_LONGARG;
      }
      else if (!strcmp(type, "unsignedlong")) {
         p->type = G__TMPLT_ULONGARG;
      }
      else if (!strcmp(type, "unsigned")) {
         fpos_t pos;
         int linenum;
         fgetpos(G__ifile.fp, &pos);
         linenum = G__ifile.line_number;
         c = G__fgetname(name, ",>=");
         if (!strcmp(name, "char")) {
            p->type = G__TMPLT_UCHARARG;
         }
         else if (!strcmp(name, "short")) {
            p->type = G__TMPLT_USHORTARG;
         }
         else if (!strcmp(name, "int")) {
            p->type = G__TMPLT_UINTARG;
         }
         else if (!strcmp(name, "long")) {
            p->type = G__TMPLT_ULONGARG;
            fgetpos(G__ifile.fp, &pos);
            linenum = G__ifile.line_number;
            c = G__fgetname(name, ",>=");
            if (!strcmp(name, "int")) {
               p->type = G__TMPLT_ULONGARG;
            }
            else {
               p->type = G__TMPLT_ULONGARG;
               fsetpos(G__ifile.fp, &pos);
               G__ifile.line_number = linenum;
            }
         }
         else {
            p->type = G__TMPLT_UINTARG;
            fsetpos(G__ifile.fp, &pos);
            G__ifile.line_number = linenum;
         }
      }
      else if (!strcmp(type, "size_t")) { // FIXME: This is a cint extension!
         p->type = G__TMPLT_SIZEARG;
      }
      else if (!strcmp(type, "float")) { // FIXME: This is a cint extension!
         p->type = G__TMPLT_FLOATARG;
      }
      else if (!strcmp(type, "double")) { // FIXME: This is a cint extension!
         p->type = G__TMPLT_DOUBLEARG;
      }
      else {
         if (G__dispsource) {
            G__fprinterr(G__serr, "Limitation: template argument type '%s' may cause problem", type);
            G__printlinenum();
         }
         p->type = G__TMPLT_INTARG;
      }
      //
      //  Now get the parameter name, if there is one.
      //
      //  template<class T,class E,int S> ...
      //                 ^
      c = G__fgetstream(name, ",>="); // FIXME: G__fgetstream_tmplt() ???
      //
      //  FIXME: We need to accumulate a reference qualifier here, if any.
      //
      //--
      //
      //  Accumulate pointer qualifiers from a type template parameter name.
      //
      while (name[0] && (name[strlen(name)-1] == '*')) { // Parameter type is pointer qualified.
         if (p->type == G__TMPLT_CLASSARG) {
            p->type = G__TMPLT_POINTERARG1;
         }
         else {
            p->type += G__TMPLT_POINTERARG1;
         }
         name[strlen(name)-1] = '\0';
      }
      p->string = (char*) malloc(strlen(name) + 1);
      strcpy(p->string, name);
      //
      //  Now get the parameter default if there is one.
      //
      //  template<class T=int,class E,int S> ...
      //                   ^
      p->default_parameter = 0;
      if (c == '=') {
         c = G__fgetstream_template(name, ",>"); // FIXME: G__fgetstream_tmplt() ???
         p->default_parameter = (char*) malloc(strlen(name) + 1);
         strcpy(p->default_parameter, name);
      }
      //
      //  c is now either "," or ">".
      //
      //  template<class T,class E,int S> ...
      //                   ^
   }
   //  template<class T,class E,int S> ...
   //                                 ^
   return tmpl_arg_list;
}

//______________________________________________________________________________
static void G__explicit_template_specialization()
{
   // Read and parse an explicit template specialization.
   //
   // The parse state on entry is:
   //
   //      template<> class A<int> { A(A& x); A& operator=(A& x); };
   //      template<> void A<int>::A(A& x) { }
   //                ^
   //--
   G__StrBuf buf_sb(G__ONELINE);
   char* buf = buf_sb;
   int cin;
   // store file position
   fpos_t store_pos;
   int store_line = G__ifile.line_number;
   fgetpos(G__ifile.fp, &store_pos);
   G__disp_mask = 1000;
   // forward proving
   cin = G__fgetname_template(buf, ":{;");
   //
   //  Handle a possible class template specialization.
   //
   if (!strcmp(buf, "class") || !strcmp(buf, "struct")) { // FIXME: We must support "union" too!
      // template<>  class A<int> { A(A& x); A& operator=(A& x); };
      //                  ^
      char* pp;
      G__StrBuf templatename_sb(G__ONELINE);
      char* templatename = templatename_sb;
      int npara = 0;
      ::Reflex::Scope envtagnum = G__get_envtagnum();
      G__Charlist call_para;
      fpos_t posend;
      int lineend;
      call_para.string = 0;
      call_para.next = 0;
      cin = G__fgetname_template(buf, ":{;");
      strcpy(templatename, buf);
      pp = strchr(templatename, '<');
      if (pp) {
         *pp = 0;
      }
      if (cin == ':') {
         cin = G__fignorestream("{;");
      }
      if (cin == '{') {
         G__disp_mask = 1;
         fseek(G__ifile.fp, -1, SEEK_CUR);
         cin = G__fignorestream("};");
      }
      fgetpos(G__ifile.fp, &posend);
      lineend = G__ifile.line_number;
      // rewind file position
      // template<> class A<int> { ... }
      //           ^--------------
      G__disp_mask = 0;
      fsetpos(G__ifile.fp, &store_pos);
      G__ifile.line_number = store_line;
      G__replacetemplate(templatename, buf, &call_para, G__ifile.fp, G__ifile.line_number, G__ifile.filenum, &store_pos, 0, 1, npara, G__get_tagnum(envtagnum));
      fsetpos(G__ifile.fp, &posend);
      G__ifile.line_number = lineend;
      return;
   }
   //
   //  It is not a class template specialization, so it
   //  must be a function template specialization.
   //
   //  Read the rest as and ordinary function definition.
   //
   //  FIXME: This is really wrong, we have not done any template argument substitution!
   //
   G__disp_mask = 0;
   fsetpos(G__ifile.fp, &store_pos);
   G__ifile.line_number = store_line;
   int brace_level = 0;
   G__exec_statement(&brace_level);
}

//______________________________________________________________________________
int Cint::Internal::G__createtemplateclass(char* new_name, G__Templatearg* targ, int isforwarddecl)
{
   // Make an entry in the table of class templates.
   //
   // Note: Parsing function, worker for  G__declare_template.
   //
   // Our parse state is one of:
   //
   //      template<class T,class E,int S> class A { .... };
   //                                     ^
   //
   //      template<class T,class E,int S> class A<B> { .... };
   //                                     ^
   //
   // or if isforwarddecl is true:
   //
   //      template<class T,class E,int S> class A;
   //                                     ^
   //
   //      template<class T,class E,int S> class A<B>;
   //                                     ^
   //
   // and we have peeked ahead to get the template id.
   //
   //--
   //
   //  Read any specialization arguments given.
   //
   G__Templatearg* spec_arg = 0;
   char* spec = strchr(new_name, '<');
   if (spec) {
      *spec = 0; // Remove the specialization args from the name.
      spec_arg = G__read_specializationarg(targ, spec + 1); // Parse the spec args into a list.
   }
   //
   //  Search for any previous declaration
   //  of the same class template.
   //
   int hash = 0;
   int i = 0;
   G__hash(new_name, hash, i)
   ::Reflex::Scope env_tagnum = G__get_envtagnum();
   int override = 0; // Are we overriding a forward declaration or precompiled definition?
   G__Definedtemplateclass* deftmpclass = &G__definedtemplateclass;
   for ( ; deftmpclass->next; deftmpclass = deftmpclass->next) {
      if ( // Name and scope match.
         (deftmpclass->hash == hash) &&
         !strcmp(deftmpclass->name, new_name) &&
         (env_tagnum == G__Dict::GetDict().GetScope(deftmpclass->parent_tagnum))
      ) { // Name and scope match.
         if (!deftmpclass->isforwarddecl && deftmpclass->def_fp) { // Possible duplicate definition.  FIXME: Why are compiled templates excluded from check?
            if (isforwarddecl) { // Duplicate is a forward declaration, ignore it and exit.
               // Ignore a forward declaration which comes lexically after a definition.
               G__fignorestream(";");
               return 0;
            }
            if (spec_arg) { // We are creating a specialization, never override, always a match.
               if (!deftmpclass->specialization) { // No specializations exist yet, create and initialize one.
                  deftmpclass->specialization = new G__Definedtemplateclass;
                  deftmpclass = deftmpclass->specialization;
                  // Initialize the new specialization.
                  deftmpclass->def_para = 0;
                  deftmpclass->next = 0;
                  deftmpclass->name = 0;
                  deftmpclass->hash = 0;
                  deftmpclass->memfunctmplt.next = 0;
                  deftmpclass->def_fp = 0;
                  deftmpclass->isforwarddecl = 0;
                  deftmpclass->instantiatedtagnum = 0;
                  deftmpclass->specialization = 0;
                  deftmpclass->spec_arg = 0;
               }
               else { // There are specializations already, move to the end of the list. // FIXME: We do not check for duplicate specialization!
                  deftmpclass = deftmpclass->specialization;
                  while (deftmpclass->next) {
                     deftmpclass = deftmpclass->next;
                  }
               }
               deftmpclass->spec_arg = spec_arg;
               override = 0; // We are not overriding.
               break; // We have a match.
            }
            //
            //  Ignore duplicate template class definition, and return.
            //
            if (G__dispmsg >= G__DISPWARN) {
               G__fprinterr(G__serr, "Warning: template %s duplicate definition", new_name);
               G__printlinenum();
            }
            G__fignorestream(";");
            return 0;
         }
         override = 1; // Flag that we will override the forward declaration or the existing precompiled definition.
         break;
      }
   }
   //
   //  If we are not overriding an old declaration/definition,
   //  then set the name.
   //
   if (!override) {
      // Set the class template name, notice that this contains the specialization arguments (as originally spelled, BAD!). // FIXME: We must make the specialization arguments into fully-qualified names.
      deftmpclass->name = (char*) malloc(strlen(new_name) + 1);
      strcpy(deftmpclass->name, new_name);
      deftmpclass->hash = hash;
   }
   //
   //  Set the parent scope.
   //
   {
      ::Reflex::Scope parent;
      if (!G__def_tagnum.IsTopScope()) {
         if (G__tagdefining != G__def_tagnum) {
            parent = G__tagdefining;
         }
         else {
            parent = G__def_tagnum;
         }
      }
      deftmpclass->parent_tagnum = G__get_tagnum(parent);
      if (!deftmpclass->parent_tagnum) {
         deftmpclass->parent_tagnum = -1;
      }
   }
   //
   //  Set the template parameter list.
   //
   if (!override || !deftmpclass->def_para) {
      deftmpclass->def_para = targ;
   }
   else {
      G__Templatearg* t1 = deftmpclass->def_para;
      G__Templatearg* t2 = targ;
      while (t1 && t2) {
         if (strcmp(t1->string, t2->string)) {
            char* tmp = t2->string;
            t2->string = t1->string;
            t1->string = tmp;
         }
         if (t1->default_parameter && t2->default_parameter) {
            G__genericerror("Error: Redefinition of default template argument");
         }
         else if (!t1->default_parameter && t2->default_parameter) {
            t1->default_parameter = t2->default_parameter;
            t2->default_parameter = 0;
         }
         t1 = t1->next;
         t2 = t2->next;
      }
      G__freetemplatearg(targ);
   }
   //
   //   Set the file pointer, line number and position.
   //
   deftmpclass->def_fp = G__ifile.fp;
   if (G__ifile.fp) {
      fgetpos(G__ifile.fp, &deftmpclass->def_pos);
   }
   deftmpclass->line = G__ifile.line_number;
   deftmpclass->filenum = G__ifile.filenum;
   //
   //  If we are not overriding a previous
   //  declaration or definition, then preallocate
   //  and initialize a new table entry at the end
   //  for next time (this is also the end marker).
   //
   if (!override) {
      deftmpclass->next = new G__Definedtemplateclass;
      deftmpclass->next->def_para = 0;
      deftmpclass->next->next = 0;
      deftmpclass->next->name = 0;
      deftmpclass->next->hash = 0;
      deftmpclass->next->memfunctmplt.next = 0;
      deftmpclass->next->def_fp = 0;
      deftmpclass->next->isforwarddecl = 0;
      deftmpclass->next->instantiatedtagnum = 0;
      deftmpclass->next->specialization = 0;
      deftmpclass->next->spec_arg = 0;
   }
   //
   //  Skip to the end of the declaration or definition.
   //
   if (targ) {
      G__fignorestream(";");
   }
   //  The parsing state is now:
   //
   //       template<class T,class E,int S> class A { .... };
   //                                                        ^
   //--
   //
   // FIXME: This is forbidden by the standard which requires
   //        that a definition have been seen when an instantiation is requested.
   //        The problem is that we do not make a distinction in G__defined_typename
   //        between just needing a declaration or needing a definition, so it always
   //        requests an instantiation, hence this crap.
   //
   // forward declaration of template -> instantiation ->
   // definition of template NOW instantiate forward declaration
   //
   if (
      (deftmpclass->isforwarddecl == 1) && // class template was previously forward declared, and
      !isforwarddecl && // this is *not* a forward declaration, and
      deftmpclass->instantiatedtagnum // we have pending instantiation requests
   ) {
      G__instantiate_templateclasslater(deftmpclass);
   }
   deftmpclass->isforwarddecl = isforwarddecl; // Flag whether or not this is a forward declaration.
   return 0;
}

//______________________________________________________________________________
static void G__instantiate_templateclasslater(G__Definedtemplateclass* deftmpclass)
{
   // Instantiation of forward declared class template when the definition is finally parsed.
   //
   // FIXME: This is forbidden by the standard which requires
   //        that a definition have been seen when an instantiation is requested.
   //        The problem is that we do not make a distinction in G__defined_typename
   //        between just needing a declaration or needing a definition, so it always
   //        requests an instantiation, hence this crap.
   //
   // forward declaration of template -> instantiation ->
   // definition of template NOW instantiate forward declaration
   //
   ::Reflex::Scope store_def_tagnum = G__def_tagnum;
   ::Reflex::Scope store_tagdefining = G__tagdefining;
   int store_def_struct_member = G__def_struct_member;
   G__StrBuf tagname_sb(G__LONGLINE);
   char* tagname = tagname_sb;
   G__IntList* ilist = deftmpclass->instantiatedtagnum;
   for ( ; ilist; ilist = ilist->next) {
      strcpy(tagname, G__struct.name[ilist->i]);
      if (G__struct.parent_tagnum[ilist->i] != -1) {
         G__def_tagnum = G__Dict::GetDict().GetScope(G__struct.parent_tagnum[ilist->i]);
         G__tagdefining = G__Dict::GetDict().GetScope(G__struct.parent_tagnum[ilist->i]);
         G__def_struct_member = 1;
      }
      else {
         G__def_tagnum = store_def_tagnum;
         G__tagdefining = store_tagdefining;
         G__def_struct_member = store_def_struct_member;
      }
      G__instantiate_templateclass(tagname, 0); // Instantiate with error messages.
   }
   G__def_tagnum = store_def_tagnum;
   G__tagdefining = store_tagdefining;
   G__def_struct_member = store_def_struct_member;
}

//______________________________________________________________________________
static int G__createtemplatefunc(char* funcname, G__Templatearg* targ, int line_number, fpos_t* ppos)
{
   // Create template function entry
   //  template<class T,class E> type func(T a,E b,int a) {}
   //                                      ^
   // Note: Parsing function, worker for G__declare_template.
   //
   G__Definetemplatefunc* deftmpfunc;
   G__StrBuf paraname_sb(G__MAXNAME);
   char* paraname = paraname_sb;
   G__StrBuf temp_sb(G__LONGLINE);
   char* temp = temp_sb;
   int tmp;
   int c;
   int pointlevel;
   int reftype;
   int unsigned_flag;
   int tagnum;
   ::Reflex::Type typenum;
   int narg;
   //
   //  Get to the end of the list.
   //
   deftmpfunc = &G__definedtemplatefunc;
   while (deftmpfunc->next) {
      deftmpfunc = deftmpfunc->next;
   }
   //
   // store linenumber , file pointer and file position
   //
   deftmpfunc->line = line_number;
   deftmpfunc->def_pos = *ppos;
   deftmpfunc->def_fp = G__ifile.fp;
   deftmpfunc->filenum = G__ifile.filenum;
   //
   // store template argument list
   //
   deftmpfunc->def_para = targ;
   //
   // store funcname and hash
   //
   {
      deftmpfunc->name = (char*) malloc(strlen(funcname) + 1);
      strcpy(deftmpfunc->name, funcname);
      char* p = G__strrstr(deftmpfunc->name, "::");
      if (p) {
         *p = 0;
         deftmpfunc->parent_tagnum = G__defined_tagname(deftmpfunc->name, 0);
         if (!deftmpfunc->parent_tagnum) {
            deftmpfunc->parent_tagnum = -1;
         }
         p = G__strrstr(funcname, "::");
         strcpy(deftmpfunc->name, p + 2);
         G__hash(deftmpfunc->name, deftmpfunc->hash, tmp);
      }
      else {
         strcpy(deftmpfunc->name, funcname);
         G__hash(funcname, deftmpfunc->hash, tmp);
         deftmpfunc->parent_tagnum = G__get_tagnum(G__get_envtagnum());
         if (!deftmpfunc->parent_tagnum) {
            deftmpfunc->parent_tagnum = -1;
         }
      }
   }
   deftmpfunc->friendtagnum = G__get_tagnum(G__friendtagnum);
   //
   //  allocate next list entry
   //
   deftmpfunc->next = new G__Definetemplatefunc;
   deftmpfunc->next->next = 0;
   deftmpfunc->next->def_para = 0;
   deftmpfunc->next->name = 0;
   for (int i = 0; i < G__MAXFUNCPARA; ++i) {
      deftmpfunc->next->func_para.ntarg[i] = 0;
      deftmpfunc->next->func_para.nt[i] = 0;
   }
   //
   //  Parse template function parameter information
   //
   Reflex::Scope store_def_tagnum = G__def_tagnum;
   Reflex::Scope store_tagdefining = G__tagdefining;
   G__def_tagnum = G__Dict::GetDict().GetScope(deftmpfunc->parent_tagnum); // for A::f(B) where B is A::B
   G__tagdefining = G__def_tagnum;
   //  template<class T,class E> type func(T a,E b,int a) {}
   //                                      ^
   tmp = 0;
   deftmpfunc->func_para.paran = 0;
   c = 0;
   // read file and get type of parameter
   while (c != ')') {
      // initialize template function parameter attributes
      deftmpfunc->func_para.type[tmp] = 0;
      deftmpfunc->func_para.tagnum[tmp] = -1;
      //deftmpfunc->func_para.typenum[tmp] = -1;
      deftmpfunc->func_para.reftype[tmp] = G__PARANORMAL;
      deftmpfunc->func_para.paradefault[tmp] = 0;
      deftmpfunc->func_para.argtmplt[tmp] = -1;
      deftmpfunc->func_para.ntarg[tmp] = 0;
      deftmpfunc->func_para.nt[tmp] = 0;
      pointlevel = 0;
      reftype = 0;
      unsigned_flag = 0;
      //  template<class T,template<class U> class E> type func(T a,E<T> b) { }
      //                                                        ^   ^
      do { // read typename
         c = G__fgetname_template(paraname, ",)<*&=");
      }
      while (
         !strcmp(paraname, "class") ||
         !strcmp(paraname, "struct") ||
         !strcmp(paraname, "const") ||
         !strcmp(paraname, "volatile") ||
         !strcmp(paraname, "typename")
      );
      // Don't barf on an empty arg list.
      if (!paraname[0] && (c == ')') && !tmp) {
         break;
      }
      //  template<class T,template<class U> class E> type func(T a,E<T> b) { }
      //                                                         ^   ^
      //  template<class T,template<class U> class E> type func(T a,E<T> b) { }
      //                                                          ^  ^
      // 1. function parameter, fixed fundamental type
      if (!strcmp(paraname, "unsigned")) {
         unsigned_flag = -1;
         if ((c != '*') && (c != '&')) {
            c = G__fgetname(paraname, ",)*&=");
         }
      }
      else if (!strcmp(paraname, "signed")) {
         unsigned_flag = 0;
         if ((c != '*') && (c != '&')) {
            c = G__fgetname(paraname, ",)*&=");
         }
      }
      if (!strcmp(paraname, "int")) {
         deftmpfunc->func_para.type[tmp] = 'i' + unsigned_flag;
      }
      else if (!strcmp(paraname, "char")) {
         deftmpfunc->func_para.type[tmp] = 'c' + unsigned_flag;
      }
      else if (!strcmp(paraname, "short")) {
         deftmpfunc->func_para.type[tmp] = 's' + unsigned_flag;
      }
      else if (!strcmp(paraname, "bool")) {
         deftmpfunc->func_para.type[tmp] = 'g';
      }
      else if (!strcmp(paraname, "long")) {
         deftmpfunc->func_para.type[tmp] = 'l' + unsigned_flag;
         if ((c != '*') && (c != '&')) {
            c = G__fgetname(paraname, ",)*&[=");
            if (!strcmp(paraname, "double")) {
               deftmpfunc->func_para.type[tmp] = 'd';
            }
         }
      }
      else if (!strcmp(paraname, "double")) {
         deftmpfunc->func_para.type[tmp] = 'd';
      }
      else if (!strcmp(paraname, "float")) {
         deftmpfunc->func_para.type[tmp] = 'f';
      }
      else if (!strcmp(paraname, "void")) {
         deftmpfunc->func_para.type[tmp] = 'y';
      }
      else if (!strcmp(paraname, "FILE")) {
         deftmpfunc->func_para.type[tmp] = 'e';
      }
      else if (unsigned_flag) {
         deftmpfunc->func_para.type[tmp] = 'i' + unsigned_flag;
      }
      else if (c == '<') {
         // 2. function parameter, template class
         char* ntargc[20];
         int ntarg[20];
         int nt = 0;
         // f(T<E,K> a) or f(c<E,K> a) or f(c<E,b> a)
         // f(T<E> a) or f(c<T> a) or f(T<c> a)
         deftmpfunc->func_para.type[tmp] = 'u';
         deftmpfunc->func_para.argtmplt[tmp] = -1;
         // deftmpfunc->func_para.typenum[tmp] = -1;
         deftmpfunc->func_para.tagnum[tmp] = -1;
         // 2.1.   f(T<x,E,y> a)
         //  ntarg   0 1 2 3
         do {
            ntarg[nt] = G__istemplatearg(paraname, deftmpfunc->def_para);
            if (!ntarg[nt]) {
               G__Definedtemplateclass* deftmpclass = G__defined_templateclass(paraname);
               if (deftmpclass && (deftmpclass->parent_tagnum != -1)) {
                  const char* parent_name = G__fulltagname(deftmpclass->parent_tagnum, 1);
                  ntargc[nt] = (char*) malloc(strlen(parent_name) + strlen(deftmpclass->name) + 3);
                  strcpy(ntargc[nt], parent_name);
                  strcat(ntargc[nt], "::");
                  strcat(ntargc[nt], deftmpclass->name);
               }
               else {
                  ntargc[nt] = (char*) malloc(strlen(paraname) + 1);
                  strcpy(ntargc[nt], paraname);
               }
            }
            ++nt;
            c = G__fgetstream(paraname, ",>");
         }
         while (c == ',');
         if (c == '>') {
            ntarg[nt] = G__istemplatearg(paraname, deftmpfunc->def_para);
            if (!ntarg[nt]) {
               G__Definedtemplateclass* deftmpclass = G__defined_templateclass(paraname);
               if (deftmpclass && (deftmpclass->parent_tagnum != -1)) {
                  const char* parent_name = G__fulltagname(deftmpclass->parent_tagnum, 1);
                  ntargc[nt] = (char*) malloc(strlen(parent_name) + strlen(deftmpclass->name) + 3);
                  strcpy(ntargc[nt], parent_name);
                  strcat(ntargc[nt], "::");
                  strcat(ntargc[nt], deftmpclass->name);
               }
               else {
                  ntargc[nt] = (char*) malloc(strlen(paraname) + 1);
                  strcpy(ntargc[nt], paraname);
               }
            }
            ++nt;
         }
         deftmpfunc->func_para.nt[tmp] = nt;
         deftmpfunc->func_para.ntarg[tmp] = (int*) malloc(sizeof(int) * nt);
         deftmpfunc->func_para.ntargc[tmp] = (char**) malloc(sizeof(char*) * nt);
         for (int i = 0; i < nt; ++i) {
            deftmpfunc->func_para.ntarg[tmp][i] = ntarg[i];
            deftmpfunc->func_para.ntargc[tmp][i] = 0;
            if (!ntarg[i]) {
               deftmpfunc->func_para.ntargc[tmp][i] = ntargc[i];
            }
         }
      }
      else if ((narg = G__istemplatearg(paraname, deftmpfunc->def_para))) {
         // 3. function parameter, template argument
         // f(T a)
         if (c == '*') {
            deftmpfunc->func_para.type[tmp] = 'U';
         }
         else {
            deftmpfunc->func_para.type[tmp] = 'u';
         }
         deftmpfunc->func_para.argtmplt[tmp] = narg;
      }
      else {
         // 4. function parameter, fixed typedef or class,struct
         // f(c a)
         // 4.1. function parameter, fixed typedef
         typenum = G__find_typedef(paraname);
         if (typenum) {
            deftmpfunc->func_para.type[tmp] = G__get_type(typenum);
            deftmpfunc->func_para.typenum[tmp] = typenum;
            deftmpfunc->func_para.tagnum[tmp] = G__get_tagnum(typenum);
         }
         // 4.2. function parameter, fixed class,struct
         else if ((tagnum = G__defined_tagname(paraname, 0)) != -1) {
            if (c == '*') {
               deftmpfunc->func_para.type[tmp] = 'U';
            }
            else {
               deftmpfunc->func_para.type[tmp] = 'u';
            }
            // deftmpfunc->func_para.typenum[tmp] = -1;
            deftmpfunc->func_para.tagnum[tmp] = tagnum;
         }
         else {
            G__genericerror("Internal error: global function template arg type");
         }
      }
      // Check pointlevel and reftype
      while ((c != ',') && (c != ')')) {
         switch (c) {
            case '(': // pointer to function
               deftmpfunc->func_para.type[tmp] = 'Y';
               // deftmpfunc->func_para.typenum[tmp] = -1;
               deftmpfunc->func_para.tagnum[tmp] = -1;
               c = G__fignorestream(")");
               c = G__fignorestream(",)");
               break;
            case '=':
               deftmpfunc->func_para.paradefault[tmp] = 1;
               c = G__fignorestream(",)");
               break;
            case '[':
               c = G__fignorestream("]");
               c = G__fgetname(temp, ",()*&[=");
               ++pointlevel;
               break;
            case '*':
               ++pointlevel;
               c = G__fgetname(temp, ",()*&[=");
               break;
            case '&':
               ++reftype;
               c = G__fgetname(temp, ",()*&[=");
               break;
            default:
               c = G__fgetname(temp, ",()*&[=");
               break;
         }
      }
      //  template<class T,template<class U> class E> type func(T a,E<T> b) {}
      //                                                           ^      ^
      if (reftype) {
         if (pointlevel) {
            deftmpfunc->func_para.type[tmp] = toupper(deftmpfunc->func_para.type[tmp]);
         }
         deftmpfunc->func_para.reftype[tmp] = G__PARAREFERENCE;
      }
      else {
         switch (pointlevel) {
            case 0:
               deftmpfunc->func_para.reftype[tmp] = G__PARANORMAL;
               break;
            case 1:
               deftmpfunc->func_para.type[tmp] = toupper(deftmpfunc->func_para.type[tmp]);
               deftmpfunc->func_para.reftype[tmp] = G__PARANORMAL;
               break;
            case 2:
               deftmpfunc->func_para.type[tmp] = toupper(deftmpfunc->func_para.type[tmp]);
               deftmpfunc->func_para.reftype[tmp] = G__PARAP2P;
               break;
            default:
               deftmpfunc->func_para.type[tmp] = toupper(deftmpfunc->func_para.type[tmp]);
               deftmpfunc->func_para.reftype[tmp] = G__PARAP2P2P;
               break;
         }
      }
      ++tmp;
      deftmpfunc->func_para.paran = tmp;
   }
   G__def_tagnum = store_def_tagnum;
   G__tagdefining = store_tagdefining;
   // Hack by Scott Snyder: try not to gag on forward decl of template memfunc
   {
      int ch = G__fignorestream(";{");
      if (ch != ';') {
         G__fignorestream("}");
      }
   }
   return 0;
}

//______________________________________________________________________________
static int G__istemplatearg(char* paraname, G__Templatearg* def_para)
{
   // search matches for template argument
   int result = 1;
   while (def_para) {
      if (!strcmp(def_para->string, paraname)) {
         return result;
      }
      def_para = def_para->next;
      ++result;
   }
   return 0;
}

//______________________________________________________________________________
static int G__createtemplatememfunc(char* new_name)
{
   // template<class T,class E,int S> type A<T,E,S>::f() { .... }
   //                                        ^
   // Note: Parsing function, worker for G__declare_template.
   //
   G__Definedtemplateclass* deftmpclass;
   G__Definedtemplatememfunc* deftmpmemfunc;
   int os = 0;
   // funcname="*f()" "&f()"
   while ((new_name[os] == '*') || (new_name[os] == '&')) {
      ++os;
   }
   // get defined tempalte class identity
   deftmpclass = G__defined_templateclass(new_name + os);
   if (!deftmpclass) {
      // error
      G__fprinterr(G__serr, "Error: Template class %s not defined", new_name + os);
      G__genericerror(0);
   }
   else {
      // get to the end of defined member function list
      deftmpmemfunc = &deftmpclass->memfunctmplt;
      while (deftmpmemfunc->next) {
         deftmpmemfunc = deftmpmemfunc->next;
      }
      // allocate member function template list
      deftmpmemfunc->next = new G__Definedtemplatememfunc;
      deftmpmemfunc->next->next = 0;
      // set file position
      deftmpmemfunc->def_fp = G__ifile.fp;
      deftmpmemfunc->line = G__ifile.line_number;
      deftmpmemfunc->filenum = G__ifile.filenum;
      fgetpos(G__ifile.fp, &deftmpmemfunc->def_pos);
      // If member function is defined after template class
      // instantiation instantiate member functions here.
      if (deftmpclass->instantiatedtagnum) {
         G__instantiate_templatememfunclater(deftmpclass, deftmpmemfunc);
      }
   }
   return 0;
}

//______________________________________________________________________________
static void G__instantiate_templatememfunclater(G__Definedtemplateclass* deftmpclass, G__Definedtemplatememfunc* deftmpmemfunc)
{
   // instantiation of forward declared template class member function
   G__IntList* ilist = deftmpclass->instantiatedtagnum;
   G__Charlist call_para;
   G__StrBuf templatename_sb(G__LONGLINE);
   char* templatename = templatename_sb;
   G__StrBuf tagname_sb(G__LONGLINE);
   char* tagname = tagname_sb;
   char* arg = 0;
   int npara = 0;
   ::Reflex::Scope store_def_tagnum = G__def_tagnum;
   ::Reflex::Scope store_tagdefining = G__tagdefining;
   int store_def_struct_member = G__def_struct_member;
   while (ilist) {
      G__ASSERT(0 <= ilist->i);
      if (!G__struct.name[ilist->i]) {
         ilist = ilist->next;
         continue;
      }
      strcpy(tagname, G__struct.name[ilist->i]);
      strcpy(templatename, tagname);
      arg = strchr(templatename, '<');
      if (arg) {
         *arg = '\0';
         ++arg;
      }
      else {
         static char cnull[1] = "";
         arg = cnull;
      }
      call_para.string = 0;
      call_para.next = 0;
      //858//G__gettemplatearglist(arg, &call_para, deftmpclass->def_para, &npara, -1, deftmpclass->specialization);
      G__gettemplatearglist(arg, &call_para, deftmpclass->def_para, &npara, -1);
      if (G__struct.parent_tagnum[ilist->i] != -1) {
         G__def_tagnum = G__Dict::GetDict().GetScope(G__struct.parent_tagnum[ilist->i]);
         G__tagdefining = G__Dict::GetDict().GetScope(G__struct.parent_tagnum[ilist->i]);
         G__def_struct_member = 1;
      }
      else {
         G__def_tagnum = store_def_tagnum;
         G__tagdefining = store_tagdefining;
         G__def_struct_member = store_def_struct_member;
      }
      G__replacetemplate(templatename, tagname, &call_para, deftmpmemfunc->def_fp, deftmpmemfunc->line, deftmpmemfunc->filenum, &(deftmpmemfunc->def_pos), deftmpclass->def_para, 0, npara, deftmpclass->parent_tagnum);
      G__freecharlist(&call_para);
      ilist = ilist->next;
   }
   G__def_tagnum = store_def_tagnum;
   G__tagdefining = store_tagdefining;
   G__def_struct_member = store_def_struct_member;
}

//______________________________________________________________________________
void Cint::Internal::G__freedeftemplateclass(G__Definedtemplateclass* deftmpclass)
{
   if (deftmpclass->next) {
      G__freedeftemplateclass(deftmpclass->next);
      delete deftmpclass->next;
      deftmpclass->next = 0;
   }
   if (deftmpclass->spec_arg) {
      G__freetemplatearg(deftmpclass->spec_arg);
      deftmpclass->spec_arg = 0;
   }
   if (deftmpclass->specialization) {
      G__freedeftemplateclass(deftmpclass->specialization);
      delete deftmpclass->specialization;
      deftmpclass->specialization = 0;
   }
   G__freetemplatearg(deftmpclass->def_para);
   deftmpclass->def_para = 0;
   if (deftmpclass->name) {
      free(deftmpclass->name);
      deftmpclass->name = 0;
   }
   G__freetemplatememfunc(&(deftmpclass->memfunctmplt));
   G__IntList_free(deftmpclass->instantiatedtagnum);
   deftmpclass->instantiatedtagnum = 0;
}

//______________________________________________________________________________
static void G__freetemplatememfunc(G__Definedtemplatememfunc* memfunctmplt)
{
   if (memfunctmplt->next) {
      G__freetemplatememfunc(memfunctmplt->next);
      delete memfunctmplt->next;
      memfunctmplt->next = 0;
   }
}

//______________________________________________________________________________
void Cint::Internal::G__freetemplatefunc(G__Definetemplatefunc* deftmpfunc)
{
   int i;
   if (deftmpfunc->next) {
      G__freetemplatefunc(deftmpfunc->next);
      delete deftmpfunc->next;
      deftmpfunc->next = 0;
   }
   if (deftmpfunc->def_para) {
      G__freetemplatearg(deftmpfunc->def_para);
      deftmpfunc->def_para = 0;
   }
   if (deftmpfunc->name) {
      free(deftmpfunc->name);
      deftmpfunc->name = 0;
      for (i = 0; i < G__MAXFUNCPARA; ++i) {
         if (deftmpfunc->func_para.ntarg[i]) {
            for (int j = 0; j < deftmpfunc->func_para.nt[i]; ++j) {
               if (deftmpfunc->func_para.ntargc[i][j]) {
                  free(deftmpfunc->func_para.ntargc[i][j]);
               }
            }
            free(deftmpfunc->func_para.ntargc[i]);
            deftmpfunc->func_para.ntargc[i] = 0;
            free(deftmpfunc->func_para.ntarg[i]);
            deftmpfunc->func_para.ntarg[i] = 0;
            deftmpfunc->func_para.nt[i] = 0;
         }
      }
   }
}

#if 0
{
   ////______________________________________________________________________________
   //int Cint::Internal::G__instantiate_templateclass(char* tagnamein, int noerror)
   //{
   //   // Instantiate a class template if needed.
   //   //
   //   // Note: If noerror then we print no error messages if the template does not exist.
   //   //
   //   int store_constvar = G__constvar;
   //   //
   //   //  Return if this instantiation already exists.
   //   //
   //   {
   //      int typenum = G__defined_typename(tagnamein);
   //      if (typenum != -1) {
   //         ::Reflex::Type ty = G__Dict::GetDict().GetTypedef(typenum);
   //         int intTagnum = G__get_tagnum(ty);
   //         return intTagnum;
   //      }
   //   }
   //#ifdef G__ASM
   //   G__abortbytecode(); // Bytecode cannot perform template instantiations.
   //#endif // G__ASM
   //   //
   //   //  Parse out template name and template argument list.
   //   //
   //   G__StrBuf templatename_sb(G__LONGLINE);
   //   char* templatename = templatename_sb;
   //   strcpy(templatename, tagnamein);
   //   char* tmpl_args_string = strchr(templatename, '<');
   //   if (tmpl_args_string) {
   //      *tmpl_args_string = '\0';
   //      ++tmpl_args_string;
   //   }
   //   else {
   //      tmpl_args_string = "";
   //   }
   //   //
   //   //  Collect any using directives in our enclosing scope.
   //   //
   //   G__inheritance* baseclass = 0;
   //   ::Reflex::Scope env_tagnum = G__get_envtagnum();
   //   int int_env_tagnum = G__get_tagnum(env_tagnum);
   //   if (env_tagnum && !env_tagnum.IsTopScope() && G__struct.baseclass[int_env_tagnum]->vec.size()) {
   //      baseclass = G__struct.baseclass[int_env_tagnum];
   //   }
   //   //
   //   //  Parse out the given scope and the template name.
   //   //  Lookup the given scope.
   //   //
   //   G__StrBuf atom_name_sb(G__LONGLINE);
   //   char* atom_name = atom_name_sb;
   //   ::Reflex::Scope given_scope = ::Reflex::Scope::GlobalScope();
   //   int hash = 0;
   //   int temp = 0;
   //   {
   //      strcpy(atom_name, templatename);
   //      char* patom = atom_name;
   //      //
   //      //  Advance to the final component of a scoped name.
   //      //
   //      char* p = G__find_first_scope_operator(patom);
   //      while (p) {
   //         patom = p + 2;
   //         p = G__find_first_scope_operator(patom);
   //      }
   //      //
   //      //  Get given scope, if any.
   //      //
   //      if (patom != atom_name) { // A scope was given.
   //         *(patom - 2) = 0; // terminate scope name
   //         if (strcmp(atom_name, "::")) { // not given as global scope
   //            int int_tagnum = G__defined_tagname(atom_name, 0); // Get specified scope.  WARNING: This may cause a template instantiation!
   //            given_scope = G__Dict::GetDict().GetScope(int_tagnum);
   //         }
   //         // Copy the atom name down.
   //         p = atom_name;
   //         while (*patom) {
   //            *p++ = *patom++;
   //         }
   //         *p = 0;
   //      }
   //      G__hash(atom_name, hash, temp)
   //   }
   //   //
   //   //  Search for the class template definition.
   //   //
   //   //  FIXME: Move this all to Reflex.
   //   //
   //   G__Definedtemplateclass* class_tmpl = &G__definedtemplateclass;
   //   for ( ; class_tmpl->next; class_tmpl = class_tmpl->next) {
   //      if ( // no match on name, next entry
   //         (hash != class_tmpl->hash) ||
   //         (atom_name[0] != class_tmpl->name[0]) ||
   //         strcmp(atom_name, class_tmpl->name)
   //      ) {
   //         continue;
   //      }
   //      //
   //      //  Look for ordinary scope resolution.
   //      //
   //      if (
   //         (given_scope == G__Dict::GetDict().GetScope(class_tmpl->parent_tagnum)) ||
   //         (
   //            (!given_scope || given_scope.IsTopScope()) &&
   //            (
   //               (class_tmpl->parent_tagnum == -1) ||
   //               (env_tagnum == G__Dict::GetDict().GetScope(class_tmpl->parent_tagnum))
   //            )
   //         )
   //      ) {
   //         break;
   //      }
   //      if (given_scope && given_scope.IsTopScope()) {
   //         continue;
   //      }
   //      if (baseclass) {
   //         // look for using directive scope resolution
   //         bool found = false;
   //         for (temp = 0; temp < baseclass->vec.size(); ++temp) {
   //            if (baseclass->vec[temp].basetagnum == class_tmpl->parent_tagnum) {
   //               found = true;
   //               break;
   //            }
   //         }
   //         if (found) {
   //            break;
   //         }
   //      }
   //      // look for enclosing scope resolution
   //      {
   //         bool found = false;
   //         ::Reflex::Scope env_parent_tagnum = env_tagnum;
   //         while (env_parent_tagnum && !env_parent_tagnum.IsTopScope()) {
   //            env_parent_tagnum = env_parent_tagnum.DeclaringScope();
   //            if (env_parent_tagnum == G__Dict::GetDict().GetScope(class_tmpl->parent_tagnum)) {
   //               found = true;
   //               break;
   //            }
   //            if (G__struct.baseclass[G__get_tagnum(env_parent_tagnum)]) {
   //               for (temp = 0; temp < G__struct.baseclass[G__get_tagnum(env_parent_tagnum)]->vec.size(); ++temp) {
   //                  if (G__struct.baseclass[G__get_tagnum(env_parent_tagnum)]->vec[temp].basetagnum == class_tmpl->parent_tagnum) {
   //                     found = true;
   //                     break;
   //                  }
   //               }
   //               if (found) {
   //                  break;
   //               }
   //            }
   //         }
   //         if (found) {
   //            break;
   //         }
   //      }
   //      // look in global scope (handle using declaration)
   //      {
   //         bool found = false;
   //         for (temp = 0; temp < G__globalusingnamespace.basen; ++temp) {
   //            if (G__globalusingnamespace.basetagnum[temp] == class_tmpl->parent_tagnum) {
   //               found = true;
   //               break;
   //            }
   //         }
   //         if (found) {
   //            break;
   //         }
   //      }
   //   }
   //   //
   //   //  Exit if class template not found.
   //   //
   //   if (!class_tmpl->next) { // No such template, error.
   //      if (!noerror) {
   //         G__fprinterr(G__serr, "Error: no such template %s", tagnamein);
   //         G__genericerror(0);
   //      }
   //      return -1;
   //   }
   //   //
   //   //  Class template exists but we have no file pointer
   //   //  to the definition, it must be precompiled, exit.
   //   //
   //   if (!class_tmpl->def_fp) { // No file pointer to source code of template.
   //      if (!noerror) {
   //         G__fprinterr(G__serr, "Limitation: Can't instantiate precompiled template %s", tagnamein);
   //         G__genericerror(0);
   //      }
   //      return -1;
   //   }
   //   //
   //   //  Parse the template argument string into a template argument list.
   //   //  Insert defaults for any omitted args.  Apply specialization
   //   //  arguments.  Count the number of arguments provided.
   //   //
   //   int store_templatearg_enclosedscope = G__templatearg_enclosedscope;
   //   G__templatearg_enclosedscope = 0;
   //   G__Charlist tmpl_arg_list;
   //   tmpl_arg_list.string = 0;
   //   tmpl_arg_list.next = 0;
   //   int npara = 0;
   //   //858//int defarg = G__gettemplatearglist(tmpl_args_string, &tmpl_arg_list, class_tmpl->def_para, &npara, class_tmpl->parent_tagnum, class_tmpl->specialization); // FIXME: Port the addition of the specialization args to cint5?
   //   int defarg = G__gettemplatearglist(tmpl_args_string, &tmpl_arg_list, class_tmpl->def_para, &npara, class_tmpl->parent_tagnum);
   //   if (defarg) { // We have modified or defaulted template arguments.
   //      // If the default-completed template argument list is not
   //      // identical character-by-character to the provided argument
   //      // list, then create a typedef to map the name as provided
   //      // to the actual name with all the default arguments filled in.
   //#if 0
   //         ::Reflex::Type typenum;
   //         int templatearg_enclosedscope = G__templatearg_enclosedscope;
   //         G__templatearg_enclosedscope = store_templatearg_enclosedscope;
   //         G__StrBuf tagname_sb(G__LONGLINE);
   //         char* tagname = tagname_sb;
   //         strcpy(tagname, tagnamein);
   //         std::string tmp = tagname;
   //         G__cattemplatearg(tagname, &tmpl_arg_list, npara);
   //         std::string short_name = tagname;
   //         strcpy(tagname, tmp.c_str());
   //         G__cattemplatearg(tagname, &tmpl_arg_list);
   //         ::Reflex::Scope tagnum;
   //         long intTagnum = G__defined_tagname(tagname, 1);
   //         if (intTagnum != -1) {
   //            tagnum = G__Dict::GetDict().GetScope(intTagnum);
   //         }
   //         else {
   //            tagnum = Reflex::Dummy::Scope();
   //         }
   //         G__settemplatealias(tagnamein, tagname, intTagnum, &tmpl_arg_list, class_tmpl->def_para, 0); // FIXME: No enclosedscope?
   //         ::Reflex::Scope parent_tagnum;
   //         if ((short_name != tagname) && !G__find_typedef(short_name.c_str())) {
   //            parent_tagnum = tagnum.DeclaringScope();
   //            typenum = G__declare_typedef(short_name.c_str(), 'u', G__get_tagnum(tagnum), G__PARANORMAL, 0, G__globalcomp, G__get_tagnum(parent_tagnum), false);
   //            if (defarg == 3) {
   //               G__struct.defaulttypenum[G__get_tagnum(tagnum)] = typenum; // FIXME: This should be G__get_typenum(typenum)!
   //            }
   //         }
   //#endif // 0
   //      int templatearg_enclosedscope = G__templatearg_enclosedscope;
   //      G__templatearg_enclosedscope = store_templatearg_enclosedscope;
   //      // Replace the provided name, by rewriting the template arguments
   //      // with versions that have "std::" removed, and by filling in all
   //      // of the default arguments, again with "std::" removed.
   //      G__StrBuf tagname_sb(G__LONGLINE);
   //      char* tagname = tagname_sb;
   //      strcpy(tagname, tagnamein);
   //      G__cattemplatearg(tagname, &tmpl_arg_list);
   //      //
   //      // Lookup and create, if it does not exist,
   //      // this template instantiation, which is the
   //      // canonical name.
   //      //
   //      // Note: We will call ourselves here if we must
   //      //       create the canonical instantiation.
   //      int int_tagnum = G__defined_tagname(tagname, 1);
   //      //
   //      //  Create a typedef mapping the provided name to
   //      //  the canonical name in the scope of the canonical
   //      //  name's parent.
   //      //
   //      int parent_tagnum = -1;
   //      if (templatearg_enclosedscope) {
   //         parent_tagnum = G__get_tagnum(G__get_envtagnum());
   //      }
   //      else {
   //         parent_tagnum = G__struct.parent_tagnum[int_tagnum];
   //         //::Reflex::Scope tagnum = G__Dict::GetDict().GetScope(int_tagnum);
   //         //if (!tagnum.DeclaringScope().IsTopScope()) { // We do not want parent_tagnum to be zero!
   //         //   parent_tagnum = G__get_tagnum(tagnum.DeclaringScope());
   //         //}
   //      }
   //      ::Reflex::Type typenum = G__declare_typedef(tagnamein, 'u', int_tagnum, G__PARANORMAL, 0, G__globalcomp, parent_tagnum, false);
   //      // Create a set of typedefs which map to the canonical
   //      // instantiation we just looked up or created.
   //      //
   //      // The constructed typedefs start from a base name constructed
   //      // by taking the provided name, adding in all default template
   //      // arguments, and then removing any and all "std::" from the
   //      // resulting set of template arguments.  Note that this base
   //      // name should be the same as the canonical instantiation we
   //      // just looked up or created.
   //      //
   //      // For example, given:
   //      //
   //      //      vector<std::string>
   //      //
   //      // the base name will be:
   //      //
   //      //      vector<string,allocator<string> >
   //      //
   //      // The first typedef created has only the required template
   //      // arguments, and further typedefs are created by adding the
   //      // default arguments one-by-one until one before the last
   //      // argument.
   //      //
   //      // So continuing the example we will make this set of typedefs:
   //      //
   //      //      vector<string>
   //      //
   //      // but not:
   //      //
   //      //      vector<string,allocator<string> >
   //      //
   //      // Warning: tagname is modified by this routine.
   //      //          All template arguments are rewritten with
   //      //          any and all "std::" removed and defaults
   //      //          are filled in.
   //      // FIXME: the resulting tagname has ">>" at the end instead of "> >" if the last argument or default argument ends with ">".
   //      G__settemplatealias(tagnamein, tagname, int_tagnum, &tmpl_arg_list, class_tmpl->def_para, templatearg_enclosedscope);
   //      if (defarg == 3) {
   //         G__struct.defaulttypenum[int_tagnum] = typenum;
   //      }
   //      G__freecharlist(&tmpl_arg_list);
   //      return int_tagnum;
   //   }
   //   //
   //   //  No template arguments were modified
   //   //  or defaulted.  We can instantiate
   //   //  exactly as given.
   //   //
   //   G__StrBuf tagname_sb(G__LONGLINE);
   //   char* tagname = tagname_sb;
   //   strcpy(tagname, tagnamein);
   //   if (given_scope && !given_scope.IsTopScope() || (templatename[0] == ':')) {
   //      int i = 0;
   //      char* p = strrchr(templatename, ':');
   //      while (p && *p) {
   //         templatename[i++] = *(++p);
   //      }
   //      sprintf(tagname, "%s<%s", templatename, tmpl_args_string);
   //   }
   //   // resolve template specialization
   //   if (class_tmpl->specialization) {
   //      class_tmpl = G__resolve_specialization(tmpl_args_string, class_tmpl, &tmpl_arg_list);
   //   }
   //   // store tagnum
   //   int intTagnum = G__struct.alltag;
   //   //
   //   ::Reflex::Scope store_tagdefining = G__tagdefining;
   //   ::Reflex::Scope store_def_tagnum = G__def_tagnum;
   //   G__tagdefining = G__Dict::GetDict().GetScope(class_tmpl->parent_tagnum);
   //   G__def_tagnum = G__tagdefining;
   //   //
   //   //  Perform the actual instantiation operation
   //   //  by string substituting the given template arguments
   //   //  in the template source into a temporary file and
   //   //  then parsing the result as input.
   //   //
   //   G__replacetemplate(templatename, tagname, &tmpl_arg_list, class_tmpl->def_fp, class_tmpl->line, class_tmpl->filenum, &(class_tmpl->def_pos), class_tmpl->def_para, class_tmpl->isforwarddecl ? 2 : 1, npara, class_tmpl->parent_tagnum);
   //   //
   //   //  Now instantiate all the member function templates.
   //   //
   //   //  FIXME: Member function templates should not be expanded unless they are used.
   //   //
   //   ::Reflex::Scope parent_tagnum = G__Dict::GetDict().GetScope(class_tmpl->parent_tagnum);
   //   while (parent_tagnum && !parent_tagnum.IsNamespace()) {
   //      parent_tagnum = parent_tagnum.DeclaringScope();
   //   }
   //   G__Definedtemplatememfunc* tmpl_mbrfunc = &class_tmpl->memfunctmplt;
   //   for ( ; tmpl_mbrfunc->next; tmpl_mbrfunc = tmpl_mbrfunc->next) {
   //      G__replacetemplate(templatename, tagname, &tmpl_arg_list, tmpl_mbrfunc->def_fp, tmpl_mbrfunc->line, tmpl_mbrfunc->filenum, &(tmpl_mbrfunc->def_pos), class_tmpl->def_para, 0, npara, G__get_tagnum(parent_tagnum));
   //   }
   //   //
   //   //
   //   //
   //   if (
   //      (intTagnum < G__struct.alltag) && // At least one of our attempted instantiations happened, and
   //      G__struct.name[intTagnum] && // the first one has a name, and
   //      strcmp(tagname, G__struct.name[intTagnum]) // it is not what we requested.
   //   ) {
   //      char* p1 = strchr(tagname, '<');
   //      char* p2 = strchr(G__struct.name[intTagnum], '<');
   //      if (
   //         p1 &&
   //         p2 &&
   //         ((p1 - tagname) == (p2 - G__struct.name[intTagnum])) &&
   //         !strncmp(tagname, G__struct.name[intTagnum], p1 - tagname)
   //      ) { // the first instantiation's template args match all of what was requested (so defaults must have been added)
   //         // For example, in t987, template <...,false> gets added to G__struct
   //         // but we know we want template <...,0>. We can't rename reflex types,
   //         // so we create a typedef to it.
   //         Reflex::Type origType = G__Dict::GetDict().GetType(intTagnum);
   //         if (origType) {
   //            std::string typedef_fullname = origType.DeclaringScope().Name(Reflex::SCOPED);
   //            if (typedef_fullname.length()) {
   //               typedef_fullname += "::";
   //            }
   //            typedef_fullname += tagname;
   //            //fprintf(stderr, "G__instantiate_templateclass: calling Reflex::TypedefTypeBuilder for '%s'\n", typedef_fullname.c_str());
   //            ::Reflex::Type result = ::Reflex::TypedefTypeBuilder(typedef_fullname.c_str(), origType);
   //            G__RflxProperties* prop = G__get_properties(result);
   //            if (prop) {
   //               prop->globalcomp = G__NOLINK;
   //               //fprintf(stderr, "G__instantiate_templateclass: registering typedef '%s'\n", result.Name().c_str());
   //               prop->typenum = G__Dict::GetDict().Register(result);
   //#ifdef G__TYPEDEFFPOS
   //               prop->filenum = G__ifile.filenum;
   //               prop->linenum = G__ifile.line_number;
   //#endif // G__TYPEDEFFPOS
   //               prop->tagnum = G__get_tagnum(origType.RawType()); // FIXME: I think this is wrong, I think it should be -1 always.
   //            }
   //         }
   //      }
   //      else { // no "else" in the old days, but:
   //         // we won't find it back with tagname, as it's now a typedef called "tagname".
   //         intTagnum = G__defined_tagname(tagname, 2);
   //      }
   //   }
   //   else { // no "else" in the old days, but:
   //      // we won't find it back with tagname, as it's now a typedef called "tagname".
   //      intTagnum = G__defined_tagname(tagname, 2);
   //   }
   //   if (intTagnum != -1) {
   //      if (class_tmpl->instantiatedtagnum) {
   //         G__IntList_addunique(class_tmpl->instantiatedtagnum, intTagnum);
   //      }
   //      else {
   //         class_tmpl->instantiatedtagnum = G__IntList_new(intTagnum, NULL);
   //      }
   //   }
   //   G__def_tagnum = store_def_tagnum;
   //   G__tagdefining = store_tagdefining;
   //   G__constvar = store_constvar;
   //   G__freecharlist(&tmpl_arg_list); // free template argument list
   //   return intTagnum; // return instantiated class template id
   //}
}
#endif // 0

//______________________________________________________________________________
int Cint::Internal::G__instantiate_templateclass(char* tagnamein, int noerror)
{
   // Instantiate a class template if needed.
   //
   // Note: If noerror then we print no error messages if the template does not exist.
   //
   //--
   //{
   //   char line[2048];
   //   FILE* fp = fopen("/proc/self/statm", "r");
   //   fgets(line, 2048, fp);
   //   fprintf(stderr, "\nG__instantiate_templateclass: vsz: %s", line);
   //   fclose(fp);
   //}
   //
   //  Return if this instantiation already exists.
   //
   {
      int typenum = G__defined_typename(tagnamein);
      if (typenum != -1) {
         ::Reflex::Type ty = G__Dict::GetDict().GetTypedef(typenum);
         int intTagnum = G__get_tagnum(ty);
         return intTagnum;
      }
   }
   int store_constvar = G__constvar;
#ifdef G__ASM
   G__abortbytecode(); // Bytecode cannot perform template instantiations.
#endif // G__ASM
   //
   //  Parse out template name and template argument list.
   //
   G__StrBuf templatename_sb(G__LONGLINE);
   char* templatename = templatename_sb;
   strcpy(templatename, tagnamein);
   char* tmpl_args_string = strchr(templatename, '<');
   if (tmpl_args_string) {
      *tmpl_args_string = '\0';
      ++tmpl_args_string;
   }
   else {
      static char cnull[1] = "";
      tmpl_args_string = cnull;
   }
   //
   //  Collect any using directives in our enclosing scope.
   //
   G__inheritance* baseclass = 0;
   ::Reflex::Scope env_tagnum = G__get_envtagnum();
   int int_env_tagnum = G__get_tagnum(env_tagnum);
   if (env_tagnum && !env_tagnum.IsTopScope() && !G__struct.baseclass[int_env_tagnum]->vec.empty()) {
      baseclass = G__struct.baseclass[int_env_tagnum];
   }
   //
   //  Parse out the given scope and the template name.
   //  Lookup the given scope.
   //
   G__StrBuf atom_name_sb(G__LONGLINE);
   char* atom_name = atom_name_sb;
   strcpy(atom_name, templatename);
   G__StrBuf scope_name_sb(G__LONGLINE);
   char* scope_name = scope_name_sb;
   scope_name[0] = '\0';
   ::Reflex::Scope given_scope;
   int hash = 0;
   int temp = 0;
   {
      char* patom = atom_name;
      //
      //  Advance to the final component of a scoped name.
      //
      char* p = G__find_first_scope_operator(patom);
      while (p) {
         patom = p + 2;
         p = G__find_first_scope_operator(patom);
      }
      //
      //  Get given scope, if any.
      //
      if (patom != atom_name) { // A scope was given.
         *(patom - 2) = 0; // terminate scope name
         strcpy(scope_name, atom_name);
         // Copy the atom name down.
         p = atom_name;
         while (*patom) {
            *p++ = *patom++;
         }
         *p = 0;
         if (
            !scope_name[0] ||
            !strcmp(scope_name, "::") // FIXME: This can never be true!
         ) { // the given scope is the global scope
            given_scope = ::Reflex::Scope::GlobalScope();
         }
         else { // not given as global scope
            int int_tagnum = G__defined_tagname(scope_name, 0); // Get specified scope.  WARNING: This may cause a template instantiation!
            if (int_tagnum != -1) {
               given_scope = G__Dict::GetDict().GetScope(int_tagnum);
            }
         }
      }
      G__hash(atom_name, hash, temp)
   }
   //
   //  Search for the class template definition.
   //
   //  FIXME: Move this all to Reflex.
   //
   G__Definedtemplateclass* class_tmpl = &G__definedtemplateclass;
   for ( ; class_tmpl->next; class_tmpl = class_tmpl->next) {
      if ( // no match on name, next entry
         (hash != class_tmpl->hash) ||
         (atom_name[0] != class_tmpl->name[0]) ||
         strcmp(atom_name, class_tmpl->name)
      ) {
         continue;
      }
      //
      //  Look for ordinary scope resolution.
      //
      if (
         (
            (!given_scope || given_scope.IsTopScope()) &&
            (
               (class_tmpl->parent_tagnum == -1) ||
               (env_tagnum == G__Dict::GetDict().GetScope(class_tmpl->parent_tagnum))
            )
         ) ||
         (
            (class_tmpl->parent_tagnum == -1) ?
               (!given_scope || given_scope.IsTopScope()) :
               (given_scope == G__Dict::GetDict().GetScope(class_tmpl->parent_tagnum))
         )
      ) {
         break;
      }
      if (given_scope && !given_scope.IsTopScope()) {
         continue;
      }
      if (baseclass) {
         // look for using directive scope resolution
         bool found = false;
         for (size_t temp1 = 0; temp1 < baseclass->vec.size(); ++temp1) {
            if (baseclass->vec[temp1].basetagnum == class_tmpl->parent_tagnum) {
               found = true;
               break;
            }
         }
         if (found) {
            break;
         }
      }
      // look for enclosing scope resolution
      {
         bool found = false;
         ::Reflex::Scope env_parent_tagnum = env_tagnum;
         while (env_parent_tagnum && !env_parent_tagnum.IsTopScope()) {
            env_parent_tagnum = env_parent_tagnum.DeclaringScope();
            if (env_parent_tagnum == G__Dict::GetDict().GetScope(class_tmpl->parent_tagnum)) {
               found = true;
               break;
            }
            if (G__struct.baseclass[G__get_tagnum(env_parent_tagnum)]) {
               for (size_t temp1 = 0; temp1 < G__struct.baseclass[G__get_tagnum(env_parent_tagnum)]->vec.size(); ++temp1) {
                  if (G__struct.baseclass[G__get_tagnum(env_parent_tagnum)]->vec[temp1].basetagnum == class_tmpl->parent_tagnum) {
                     found = true;
                     break;
                  }
               }
               if (found) {
                  break;
               }
            }
         }
         if (found) {
            break;
         }
      }
      // look in global scope (handle using declaration)
      {
         bool found = false;
         for (size_t temp1 = 0; temp1 < G__globalusingnamespace.vec.size(); ++temp1) {
            if (G__globalusingnamespace.vec[temp1].basetagnum == class_tmpl->parent_tagnum) {
               found = true;
               break;
            }
         }
         if (found) {
            break;
         }
      }
   }
   //
   //  Exit if class template not found.
   //
   if (!class_tmpl->next) { // No such template, error.
      if (!noerror) {
         G__fprinterr(G__serr, "Error: no such template %s", tagnamein);
         G__genericerror(0);
      }
      return -1;
   }
   //
   //  Class template exists but we have no file pointer
   //  to the definition, it must be precompiled, exit.
   //
   if (!class_tmpl->def_fp) { // No file pointer to source code of template.
      if (!noerror) {
         G__fprinterr(G__serr, "Limitation: Can't instantiate precompiled template %s", tagnamein);
         G__genericerror(0);
      }
      return -1;
   }
   //
   //  Parse the template argument string into a template argument list.
   //  Insert defaults for any omitted args.  Copy unchanged any arguments
   //  that match specialization arguments exactly.  Count the number of
   //  arguments provided.
   //
   G__Charlist* tmpl_arg_list = new G__Charlist;
   int npara = 0;
   int defarg = 0;
   int create_in_envtagnum = 0;
   {
      int store_templatearg_enclosedscope = G__templatearg_enclosedscope;
      G__templatearg_enclosedscope = 0;
      //858//defarg = G__gettemplatearglist(tmpl_args_string, tmpl_arg_list, class_tmpl->def_para, &npara, class_tmpl->parent_tagnum, class_tmpl->specialization); // FIXME: Port the addition of the specialization args to cint5?
      defarg = G__gettemplatearglist(tmpl_args_string, tmpl_arg_list, class_tmpl->def_para, &npara, class_tmpl->parent_tagnum);
      create_in_envtagnum = G__templatearg_enclosedscope;
      if (defarg) {
         G__templatearg_enclosedscope = store_templatearg_enclosedscope;
      }
   }
   //
   //  Remove any given scope qualifiers from the template name
   //  to make the simple template name.
   //
   G__StrBuf simple_templatename_sb(G__LONGLINE);
   char* simple_templatename = simple_templatename_sb;
   strcpy(simple_templatename, templatename);
   if (
      (given_scope && !given_scope.IsTopScope()) ||
      (simple_templatename[0] == ':')
   ) {
      char* p = strrchr(simple_templatename, ':');
      int i = 0;
      while (p && *p) {
         simple_templatename[i++] = *(++p);
      }
   }
   //
   //  Lookup instantiation name and return it it already exists.
   //
   {
      //
      //  Create the scoped instantiation name by rewriting the template arguments
      //  with versions that have "std::" removed, and by filling in all
      //  of the default arguments, again with "std::" removed.
      //
      G__StrBuf scoped_name_sb(G__LONGLINE);
      char* scoped_name = scoped_name_sb;
      strcpy(scoped_name, tagnamein);
      G__cattemplatearg(scoped_name, tmpl_arg_list);
      //
      //  Lookup instantiation name and return it it already exists.
      //
      if (!class_tmpl->isforwarddecl) { // We must always instantiate a forward declared template.
         //::Reflex::Scope existing_scope = ::Reflex::Scope::Lookup(scoped_name);
         //if (existing_scope) {
         //   int ret = G__get_tagnum(existing_scope);
         //   G__freecharlist(tmpl_arg_list);
         //   delete tmpl_arg_list;
         //   return ret;
         //}
         int intTagnum = G__defined_tagname(scoped_name, 2); // Try to find it, with autoloading enabled.
         if (intTagnum != -1) { // Got it, done.
            //
            //  If we succeeded and we applied default values
            //  to the provided template arguments or we
            //  expanded the typename of any of the provided
            //  template arguments, then create a typedef mapping
            //  the provided name to the canonical name in the
            //  scope of the canonical name's parent.  This is
            //  used by G__fulltagname() and G__type2string()
            //  to change the spelling of the name of the type
            //  used for calculating class checksums.  It will
            //  also be noticed at the beginning of this routine
            //  the next time it is called with the same argument,
            //  and will be used to satisfy the request directly.
            //
            if ((intTagnum != -1) && defarg) {
               G__StrBuf unscoped_tagname_sb(G__LONGLINE);
               char* unscoped_tagname = unscoped_tagname_sb;
               strcpy(unscoped_tagname, simple_templatename);
               strcat(unscoped_tagname, "<");
               strcat(unscoped_tagname, tmpl_args_string);
               int parent_tagnum = -1;
               if (create_in_envtagnum) {
                  parent_tagnum = G__get_tagnum(G__get_envtagnum());
               }
               else {
                  parent_tagnum = G__struct.parent_tagnum[intTagnum];
                  //::Reflex::Scope tagnum = G__Dict::GetDict().GetScope(intTagnum);
                  //if (!tagnum.DeclaringScope().IsTopScope()) { // We do not want parent_tagnum to be zero!
                  //   parent_tagnum = G__get_tagnum(tagnum.DeclaringScope());
                  //}
               }
               ::Reflex::Type typenum = G__declare_typedef(unscoped_tagname, 'u', intTagnum, G__PARANORMAL, 0, G__globalcomp, parent_tagnum, false);
               if (defarg == 3) {
                  G__struct.defaulttypenum[intTagnum] = typenum; // Trick G__fulltagname() and G__type2string() into printing the provided name.  Well not actually, G__OLDIMPLEMENTATION1503 is defined to turn this off in G__val2a.cxx.
               }
            }
            //
            //  Restore global state and return.
            //
            G__constvar = store_constvar;
            G__freecharlist(tmpl_arg_list);
            delete tmpl_arg_list;
            return intTagnum;
         }
      }
   }
   //
   //  We are going to instantiate, must zero this.
   //
   G__templatearg_enclosedscope = 0;
   //
   //  Attempt to use ACLiC to generate a dictionary
   //  for the requested template-id.
   //
   if (Cint::G__GetGenerateDictionary()) {
      int int_tagnum = G__generate_template_dict(tagnamein, class_tmpl, tmpl_arg_list);
      if (int_tagnum != -1) {
         //
         //  Restore global state and return.
         //
         G__constvar = store_constvar;
         G__freecharlist(tmpl_arg_list);
         delete tmpl_arg_list; 
         return int_tagnum;
      }
   }
   //
   //  The instantiation we actually create will have a name
   //  made by rewriting the template arguments with versions
   //  that have "std::" removed, and by filling in all
   //  of the default arguments, again with "std::" removed.
   //
   G__StrBuf tagname_sb(G__LONGLINE);
   char* tagname = tagname_sb;
   strcpy(tagname, simple_templatename);
   strcat(tagname, "<");
   G__cattemplatearg(tagname, tmpl_arg_list);
   //
   //  If the class template has any specializations,
   //  search to see if we match one of them, if so
   //  switch to using that specialization.
   //
   //  WARNING: If a match is found, tmpl_arg_list will be modified by
   //           having pointer, reference, and const modifiers removed
   //           that are part of the specialization arguments.
   //
   //           Also tmpl_arg_list will be modified to be appropriate
   //           for the specialized template.
   //
   G__Definedtemplateclass* primary_class_tmpl = class_tmpl;
   if (primary_class_tmpl->specialization) {
      G__StrBuf expanded_tmpl_args_string_sb(G__LONGLINE);
      char* expanded_tmpl_args_string = expanded_tmpl_args_string_sb;
      strcpy(expanded_tmpl_args_string, "<");
      G__cattemplatearg(expanded_tmpl_args_string, tmpl_arg_list);
      G__freecharlist(tmpl_arg_list);
      tmpl_arg_list->string = 0;
      tmpl_arg_list->next = 0;
      npara = 0;
      int store_templatearg_enclosedscope = G__templatearg_enclosedscope;
      G__templatearg_enclosedscope = 0;
      //858//G__gettemplatearglist(expanded_tmpl_args_string + 1, tmpl_arg_list, class_tmpl->def_para, &npara, class_tmpl->parent_tagnum, class_tmpl->specialization); // FIXME: Port the addition of the specialization args to cint5?
      G__gettemplatearglist(expanded_tmpl_args_string + 1, tmpl_arg_list, class_tmpl->def_para, &npara, class_tmpl->parent_tagnum);
      create_in_envtagnum = G__templatearg_enclosedscope;
      G__templatearg_enclosedscope = store_templatearg_enclosedscope;
      class_tmpl = G__resolve_specialization(expanded_tmpl_args_string + 1, class_tmpl, tmpl_arg_list);
   }
   //
   //  Switch current scope to declaring scope of template.
   //
   ::Reflex::Scope store_tagdefining = G__tagdefining;
   ::Reflex::Scope store_def_tagnum = G__def_tagnum;
   G__tagdefining = G__Dict::GetDict().GetScope(class_tmpl->parent_tagnum);
   G__def_tagnum = G__tagdefining;
   //
   //  Save a marker so we can know if we
   //  created any new classes.
   //
   int old_scope_size = ::Reflex::Scope::ScopeSize();
   //int old_int_tagnum = G__struct.alltag;
   //fprintf(stderr, "G__instantiate_templateclass: --------------------\n");
   //fprintf(stderr, "G__instantiate_templateclass: old name: %d '%s'\n", old_int_tagnum - 1, G__struct.name[old_int_tagnum-1]);
   //
   //  Perform the actual instantiation operation
   //  by string substituting the given template arguments
   //  in the template source into a temporary file and
   //  then parsing the result as input.
   //
   //fprintf(stderr, "G__instantiate_templateclass: templatename: '%s' tagname '%s' parent: '%s'\n", simple_templatename, tagname, (class_tmpl->parent_tagnum == -1) ? "" : G__struct.name[class_tmpl->parent_tagnum]);
   G__replacetemplate(simple_templatename, tagname, tmpl_arg_list, class_tmpl->def_fp, class_tmpl->line, class_tmpl->filenum, &class_tmpl->def_pos, class_tmpl->def_para, class_tmpl->isforwarddecl ? 2 : 1, npara, class_tmpl->parent_tagnum);
   //
   //  Get new number of classes to compare against the
   //  old number to see if we actually created any
   //  classes in the previous step.
   //
   int new_scope_size = ::Reflex::Scope::ScopeSize();
   //int new_int_tagnum = G__struct.alltag;
   //
   //  Now instantiate all the member function templates.
   //
   //  FIXME: Member function templates should not be defined unless they are used, they should just be forward declared.
   //
   //  TODO: Why are class template member function template instantiations instantiated in the enclosing namespace scope?
   {
      ::Reflex::Scope parent_tagnum = G__Dict::GetDict().GetScope(class_tmpl->parent_tagnum);
      while (parent_tagnum && !parent_tagnum.IsNamespace()) {
         parent_tagnum = parent_tagnum.DeclaringScope();
      }
      G__Definedtemplatememfunc* tmpl_mbrfunc = &class_tmpl->memfunctmplt;
      for ( ; tmpl_mbrfunc->next; tmpl_mbrfunc = tmpl_mbrfunc->next) {
         G__replacetemplate(simple_templatename, tagname, tmpl_arg_list, tmpl_mbrfunc->def_fp, tmpl_mbrfunc->line, tmpl_mbrfunc->filenum, &(tmpl_mbrfunc->def_pos), class_tmpl->def_para, 0, npara, G__get_tagnum(parent_tagnum));
      }
   }
   //
   //  Restore state.
   //
   G__def_tagnum = store_def_tagnum;
   G__tagdefining = store_tagdefining;
   //
   //  Try to find the instantiation.  It may not have been created
   //  by the substitution above because it was forward declared
   //  and merely replaced by the substitution, not created, so we
   //  cannot just take the most recently created type.
   //
   //  Note:  This lookup will fail for a template specialization that
   //         changes the tagname we tried to create it with.  This happens
   //         because we do not create the tagname correctly for specializations.
   //         FIXME: We need to fix this!  Beware, it affects class checksum calculations.
   //
   //         For example, given:
   //
   //              class A;
   //              class B;
   //              class C;
   //              class D;
   //
   //              template<class T, class U> class G {};
   //              template class G<A,B> {};
   //
   //              template<class T, class U> class H {};
   //              template<class T> class H<B,T> {};
   //
   //              G<C,D> x;
   //              H<A,int> y;
   //
   //
   //         we will request,
   //
   //              for x: G<C,D>
   //              for y: H<A,int>
   //
   //         but we will get,
   //
   //              for x: G<A,B>
   //              for y: H<B,int>
   //
   //         because we do not do specialization correctly.
   //
   int intTagnum = G__defined_tagname(tagname, 2);
   //if (new_scope_size > old_scope_size) {
   //   for (int i = old_int_tagnum; i < new_int_tagnum; ++i) {
   //      if (G__struct.name[i]) {
   //         fprintf(stderr, "G__instantiate_templateclass: new name: %d '%s'\n", i, G__struct.name[i]);
   //      }
   //      else {
   //         fprintf(stderr, "G__instantiate_templateclass: new name: %d ''\n", i);
   //      }
   //   }
   //}
   if (intTagnum != -1) { // Found.
      //fprintf(stderr, "G__instantiate_templateclass: lookup: %d '%s'\n", intTagnum, G__struct.name[intTagnum]);
   }
   else { // Defaults may have been removed by G__replacetemplate(), or specialization may have changed the name.
      //fprintf(stderr, "G__instantiate_templateclass: lookup: %d ''\n", intTagnum);
      if (new_scope_size > old_scope_size) { // There is hope, some classes have been created.
         //
         //  Not found, try first new class created.
         //
         ::Reflex::Scope new_tagnum = ::Reflex::Scope::ScopeAt(old_scope_size); // Note this is first new class declared above.
         intTagnum = G__get_tagnum(new_tagnum);
         char* p1 = strchr(tagname, '<');
         char* p2 = strchr(G__struct.name[intTagnum], '<');
         if (p1 && p2 && ((p1 - tagname) == (p2 - G__struct.name[intTagnum])) && !strncmp(tagname, G__struct.name[intTagnum], p1 - tagname)) { // We have a template name match with what was requested.
            //
            //  Create a typedef mapping the name we had intended
            //  to create to the name which was actually created.
            //  They can be different in the case where G__replacetemplate
            //  removed default arguments to shorten the name, or in the
            //  case where we instantiated a partial specialization,
            //  in which case we do not properly change tagname to reflect
            //  what will actually be created.  We should fix that.
            //
            //  Note: cint5 did this by overwriting the the G__struct.name[i] of
            //        the actually created instantiation.  We cannot do this because
            //        reflex has immutable typenames.
            //
            int parent_tagnum = -1;
            if (!new_tagnum.DeclaringScope().IsTopScope()) { // We do not want parent_tagnum to be zero!
               parent_tagnum = G__get_tagnum(new_tagnum.DeclaringScope());
            }
            ::Reflex::Type typenum = G__declare_typedef(tagname, 'u', intTagnum, G__PARANORMAL, 0, G__globalcomp, parent_tagnum, false);
            //G__struct.defaulttypenum[intTagnum] = typenum; // Trick G__fulltagname() and G__type2string() into printing the overriding name.
         }
      }
   }
   //
   //  If we succeeded and we applied default values
   //  to the provided template arguments or we
   //  expanded the typename of any of the provided
   //  template arguments, then create a typedef mapping
   //  the provided name to the canonical name in the
   //  scope of the canonical name's parent.  This is
   //  used by G__fulltagname() and G__type2string()
   //  to change the spelling of the name of the type
   //  used for calculating class checksums.  It will
   //  also be noticed at the beginning of this routine
   //  the next time it is called with the same argument,
   //  and will be used to satisfy the request directly.
   //
   if ((intTagnum != -1) && defarg) {
      G__StrBuf unscoped_tagname_sb(G__LONGLINE);
      char* unscoped_tagname = unscoped_tagname_sb;
      strcpy(unscoped_tagname, simple_templatename);
      strcat(unscoped_tagname, "<");
      strcat(unscoped_tagname, tmpl_args_string);
      int parent_tagnum = -1;
      if (create_in_envtagnum) {
         parent_tagnum = G__get_tagnum(G__get_envtagnum());
      }
      else {
         parent_tagnum = G__struct.parent_tagnum[intTagnum];
         //::Reflex::Scope tagnum = G__Dict::GetDict().GetScope(intTagnum);
         //if (!tagnum.DeclaringScope().IsTopScope()) { // We do not want parent_tagnum to be zero!
         //   parent_tagnum = G__get_tagnum(tagnum.DeclaringScope());
         //}
      }
      ::Reflex::Type typenum = G__declare_typedef(unscoped_tagname, 'u', intTagnum, G__PARANORMAL, 0, G__globalcomp, parent_tagnum, false);
      if (defarg == 3) {
         G__struct.defaulttypenum[intTagnum] = typenum; // Trick G__fulltagname() and G__type2string() into printing the provided name.  Well not actually, G__OLDIMPLEMENTATION1503 is defined to turn this off in G__val2a.cxx.
      }
   }
   //
   //  Add to the list of instantiations for the template.
   //
   if (intTagnum != -1) {
      //fprintf(stderr,"G__instantiate_templateclass: adding tagnum for %d '%s'\n", intTagnum, G__struct.name[intTagnum]);
      if (class_tmpl->instantiatedtagnum) {
         G__IntList_addunique(class_tmpl->instantiatedtagnum, intTagnum);
      }
      else {
         class_tmpl->instantiatedtagnum = G__IntList_new(intTagnum, 0);
      }
   }
   // Create a set of typedefs which map to the canonical
   // instantiation we just created.
   //
   // The constructed typedefs start from a base name constructed
   // by taking the provided name, adding in all default template
   // arguments, and then removing any and all "std::" from the
   // resulting set of template arguments.  Note that this base
   // name should be the same as the canonical instantiation we
   // just looked up or created.
   //
   // For example, given:
   //
   //      vector<std::string>
   //
   // the base name will be:
   //
   //      vector<string,allocator<string> >
   //
   // The first typedef created has only the required template
   // arguments, and further typedefs are created by adding the
   // default arguments one-by-one until one before the last
   // argument.
   //
   // So continuing the example we will make this set of typedefs:
   //
   //      vector<string>
   //
   // but not:
   //
   //      vector<string,allocator<string> >
   //
   if (intTagnum != -1) {
      if (!class_tmpl->spec_arg) {
         G__settemplatealias(scope_name, tagname, intTagnum, tmpl_arg_list, class_tmpl->def_para, create_in_envtagnum);
      }
      else {
         G__settemplatealias(scope_name, tagname, intTagnum, tmpl_arg_list, primary_class_tmpl->def_para, create_in_envtagnum);
      }
   }
   //
   //  Restore global state and return.
   //
   G__constvar = store_constvar;
   G__freecharlist(tmpl_arg_list);
   delete tmpl_arg_list;
   return intTagnum;
}

//______________________________________________________________________________
static void G__cattemplatearg(char* template_id, G__Charlist* tmpl_arg_list, int npara /*= 0*/)
{
   // Concatenate template name and template arguments.
   // if npara, use only npara template arguments.
   char* p = strchr(template_id, '<');
   if (p) {
      ++p;
   }
   else {
      p = template_id + strlen(template_id);
      *p++ = '<';
   }
   for (int i = 0; tmpl_arg_list->next && (!npara || (i < npara)); tmpl_arg_list = tmpl_arg_list->next, ++i) {
      if (i) {
         *p = ',';
         ++p;
      }
      strcpy(p, tmpl_arg_list->string);
      p += strlen(tmpl_arg_list->string);
   }
   if (*(p - 1) == '>') {
      *p = ' ';
      ++p;
   }
   *p = '>';
   ++p;
   *p = '\0';
   return;
}

//______________________________________________________________________________
static void G__settemplatealias(const char* scope_name, const char* template_id, int tagnum, G__Charlist* tmpl_arg_list, G__Templatearg* tmpl_param, int create_in_envtagnum)
{
   // Declare a set of typedefs mapping short names for the passed
   // template instantiation to the canonical name.
   //
   // Note:  This routine expects tagnamein to be fully expanded
   //        with default arguments.
   //
   G__StrBuf short_name_sb(G__LONGLINE);
   char* short_name = short_name_sb;
   strcpy(short_name, template_id);
   char* p = strchr(short_name, '<');
   if (p) {
      ++p;
   }
   else {
      p = short_name + strlen(short_name);
      *p++ = '<';
   }
   while (tmpl_arg_list->next) { // Stop before the last given template argument.
      if (tmpl_param->default_parameter) { // This arg is a default, make a short name.
         //
         //  Replace the final ',' with '>' temporarily.
         //
         char oldp = p[-1];
         char oldp2 = *(p-2);
         if (oldp == '<') { // all template args have defaults
            p[-1] = 0;
         }
         else {
            if (oldp2 == '>') {
               *(p-1) = ' ';
               ++p;
            }
            p[-1] = '>';
            *p = 0;
         }
         //
         //  If there is no typedef for this short name, make it.
         //
         G__StrBuf scoped_short_name_sb(G__LONGLINE);
         char* scoped_short_name = scoped_short_name_sb;
         strcpy(scoped_short_name, scope_name);
         if (scope_name[0]) {
            strcat(scoped_short_name, "::");
         }
         strcat(scoped_short_name, short_name);
         if (!G__find_typedef(scoped_short_name)) {
            int parent_tagnum = -1;
            if (create_in_envtagnum) {
               parent_tagnum = G__get_tagnum(G__get_envtagnum());
            }
            else {
               parent_tagnum = G__struct.parent_tagnum[tagnum];
               //::Reflex::Scope scope = G__Dict::GetDict().GetScope(tagnum);
               //if (!scope.DeclaringScope().IsTopScope()) { // We do not want parent_tagnum to be zero!
               //   parent_tagnum = G__get_tagnum(scope.DeclaringScope());
               //}
            }
            G__declare_typedef(short_name, 'u', tagnum, G__PARANORMAL, 0, G__globalcomp, parent_tagnum, false);
         }
         // Restore final ','.
         if (oldp2 == '>') {
            --p;
         }
         p[-1] = oldp;
      }
      //
      //  Add current template argument to
      //  the short name and continue.
      //
      strcpy(p, tmpl_arg_list->string);
      p += strlen(tmpl_arg_list->string);
      tmpl_arg_list = tmpl_arg_list->next;
      tmpl_param = tmpl_param->next;
      if (tmpl_arg_list->next) {
         *p = ',';
         ++p;
      } else {
         if (p[-1] == '>') {
            *p = ' ';
            ++p;
         }
      }
   }
}

//______________________________________________________________________________
static G__Definedtemplateclass* G__resolve_specialization(char* tmpl_args_string, G__Definedtemplateclass* class_tmpl, G__Charlist* expanded_tmpl_arg_list)
{
   // Attempt to find a matching specialization for the given template and the given template argument string.
   //
   // If we find a match, then modify the expanded_tmpl_arg_list by removing any
   // pointer, reference, or const qualifiers.
   //
   //--
   G__Definedtemplateclass* best_match = class_tmpl; // return value
   //
   //  Loop over all known specializations looking for a match.
   //
   int best_rating = 0;
   G__Definedtemplateclass* spec = class_tmpl->specialization;
   for ( ; spec->next; spec = spec->next) { // Search all available specializations for a best match.
      //
      //  Loop over all specialization arguments,
      //  and rate the match.
      //
      int rating = 0;
      G__Templatearg* tmpl_param = class_tmpl->def_para; // parameters of the class template
      G__Templatearg* spec_arg = spec->spec_arg; // specialization's arguments
      G__Templatearg* given_spec_arg = G__read_specializationarg(spec->def_para, tmpl_args_string); // given spec arguments
      while (spec_arg && given_spec_arg) { // FIXME: We need to match *all* of the specialization's arguments, not just some of them.
         if (given_spec_arg->type == spec_arg->type) { // Possible exact match.
            rating += 10;
            if ((tmpl_param->type & 0xff) != G__TMPLT_CLASSARG) {
               // FIXME: We do not handle template template parameters correctly here!
               // Values must match.
               std::string buf;
               buf.reserve(G__LONGLINE);
               buf = "(";
               buf += spec_arg->string;
               buf += ") != (";
               buf += given_spec_arg->string;
               buf += ")";
               int old = G__const_noerror;
               G__const_noerror = 1;
               if (G__bool(G__getexpr((char*) buf.c_str()))) {
                  if (G__security_error) {
                     G__security_error = 0;
                  }
                  else {
                     rating = 0;
                     G__const_noerror = old;
                     break;
                  }
               }
               G__const_noerror = old;
            }
            else { // This is a type template parameter.
               // Type names must match.
               if (!strstr(spec_arg->string, "//P")) { // The specialization argument does not refer to a specialization parameter.
                  if (strcmp(given_spec_arg->string, spec_arg->string)) { // Error, no type name match.
                     rating = 0;
                     break;
                  }
               }
            }
         }
         else { // check reference, pointer, and const modifiers
            if (
               ((tmpl_param->type & 0xff) != G__TMPLT_CLASSARG) &&
               ((tmpl_param->type & 0xff) != G__TMPLT_TMPLTARG) &&
               (given_spec_arg->type & 0xff) != (spec_arg->type & 0xff)
            ) { // No match, base types do not match.
               rating = 0;
               break;
            }
            int spec_r = spec_arg->type & G__TMPLT_REFERENCEARG;
            int call_r = given_spec_arg->type & G__TMPLT_REFERENCEARG;
            int spec_p = spec_arg->type & G__TMPLT_POINTERARGMASK;
            int call_p = given_spec_arg->type & G__TMPLT_POINTERARGMASK;
            int spec_c = spec_arg->type & G__TMPLT_CONSTARG;
            int call_c = given_spec_arg->type & G__TMPLT_CONSTARG;
            if (call_r == spec_r) {
               ++rating;
            }
            else if (spec_r > call_r) {
               rating = 0;
               break;
            }
            if (call_p == spec_p) {
               ++rating;
            }
            else if (spec_p > call_p) {
               rating = 0;
               break;
            }
            if (call_c == spec_c) {
               ++rating;
            }
            else if (spec_c > call_c) {
               rating = 0;
               break;
            }
         }
         tmpl_param = tmpl_param->next;
         spec_arg = spec_arg->next;
         given_spec_arg = given_spec_arg->next;
      }
      //
      //  Check if we have a new best match.
      //
      if (rating > best_rating) { // FIXME: Notice that we keep only the first best match, we need to check for ambiguous matches!
         best_rating = rating;
         best_match = spec;
      }
      G__freetemplatearg(given_spec_arg);
   }
   //
   //  If we found a matching specialization, then
   //  remove the specialization argument modifiers
   //  (pointer, reference, const) from the default-expanded
   //  given arguments, so they are not duplicated during
   //  the instantiation of the specialization.
   //
   //  Then construct a new template argument list
   //  appropriate for the selected specialization
   //  using the given arguments (which are appropriate
   //  for the primary template, but not the specialization).
   //
   if (best_match != class_tmpl) { // We found a matching specialization
      G__Templatearg* given_spec_arg = G__read_specializationarg(best_match->def_para, tmpl_args_string); // given spec arguments (which are default-expanded).
      G__modify_callpara(best_match->spec_arg, given_spec_arg, expanded_tmpl_arg_list); // Remove any spec arg modifers (pointer, reference, const) from the expanded_tmpl_arg_list so that they are not duplicated during instantiation of the specialization.
      G__freetemplatearg(given_spec_arg);
      //
      //  Construct a new template argument list
      //  appropriate for the selected specialization.
      //
      G__Charlist* new_tmpl_arg_head = new G__Charlist;
      int i = 1;
      G__Charlist* new_tmpl_arg = new_tmpl_arg_head;
      for (G__Templatearg* spec_def_para = best_match->def_para; spec_def_para; spec_def_para = spec_def_para->next) { // Create an argument for each specialization parameter.
         G__StrBuf temp_sb(G__LONGLINE);
         char* temp = temp_sb;
         sprintf(temp, "//P%d//", i);
         bool found = false;
         G__Templatearg* spec_arg = best_match->spec_arg;
         G__Charlist* expanded_tmpl_arg = expanded_tmpl_arg_list;
         for ( ; spec_arg; ) { // Find a specialization argument that matches the current specialization parameter (could be more than one).
            if (strstr(spec_arg->string, temp)) { // Found the arg corresponding to this param.
               found = true;
               if (i != 1) {
                  new_tmpl_arg->next = new G__Charlist;
                  new_tmpl_arg = new_tmpl_arg->next;
               }
               new_tmpl_arg->string = (char*) malloc(strlen(expanded_tmpl_arg->string) + 1);
               strcpy(new_tmpl_arg->string, expanded_tmpl_arg->string);
               break;
            }
            spec_arg = spec_arg->next;
            expanded_tmpl_arg = expanded_tmpl_arg->next;
         }
         if (!found) {
            fprintf(stderr, "G__resolve_specialization: Could not match specialization parameter #%d!\n", i);
            return class_tmpl; // Give up on best-match specialization.
         }
      }
      new_tmpl_arg = 0;
      G__freecharlist(expanded_tmpl_arg_list);
      if (expanded_tmpl_arg_list->string) {
         free(expanded_tmpl_arg_list->string);
      }
      expanded_tmpl_arg_list->string = (char*) malloc(strlen(new_tmpl_arg_head->string) + 1);
      strcpy(expanded_tmpl_arg_list->string, new_tmpl_arg_head->string);
      expanded_tmpl_arg_list->next = new_tmpl_arg_head->next;
      free(new_tmpl_arg_head->string);
      new_tmpl_arg_head->next = 0;
      delete (new_tmpl_arg_head);
      new_tmpl_arg_head = 0;
   }
   return best_match;
}

//______________________________________________________________________________
static void G__modify_callpara(G__Templatearg* spec_arg_in, G__Templatearg* given_spec_arg_in, G__Charlist* expanded_tmpl_arg_in)
{
   // Remove any pointer, reference, or const modifiers from the expanded_tmp_arg.
   G__Templatearg* spec_arg = spec_arg_in;
   G__Templatearg* given_spec_arg = given_spec_arg_in;
   G__Charlist* expanded_tmpl_arg = expanded_tmpl_arg_in;
   while (spec_arg && given_spec_arg && expanded_tmpl_arg) {
      int spec_p = spec_arg->type & G__TMPLT_POINTERARGMASK;
      int call_p = given_spec_arg->type & G__TMPLT_POINTERARGMASK;
      int spec_r = spec_arg->type & G__TMPLT_REFERENCEARG;
      int call_r = given_spec_arg->type & G__TMPLT_REFERENCEARG;
      int spec_c = spec_arg->type & G__TMPLT_CONSTARG;
      int call_c = given_spec_arg->type & G__TMPLT_CONSTARG;
      if ((spec_p > 0) && (spec_p <= call_p)) {
         int n = spec_p / G__TMPLT_POINTERARG1;
         char buf[10];
         for (int i = 0; i < n; ++i) {
            buf[i] = '*';
         }
         buf[n] = 0;
         G__delete_end_string(expanded_tmpl_arg->string, buf);
      }
      if (spec_r && (call_r == spec_r)) {
         G__delete_end_string(expanded_tmpl_arg->string, "&");
      }
      if (spec_c && (call_c == spec_c)) {
         G__delete_string(expanded_tmpl_arg->string, "const ");
      }
      spec_arg = spec_arg->next;
      given_spec_arg = given_spec_arg->next;
      expanded_tmpl_arg = expanded_tmpl_arg->next;
   }
   spec_arg = spec_arg_in;
   given_spec_arg = given_spec_arg_in;
   expanded_tmpl_arg = expanded_tmpl_arg_in;
}

//______________________________________________________________________________
static void G__delete_string(char* str, const char* del)
{
   char* p = strstr(str, del);
   if (p) {
      char* e = p + strlen(del);
      while (*e) {
         *(p++) = *(e++);
      }
      *p = 0;
   }
}

//______________________________________________________________________________
static void G__delete_end_string(char* str, const char* del)
{
   // Remove the last occurence of 'del' (if any)
   char* e;
   char* p = strstr(str, del);
   char* t = 0;
   while (p && (t = strstr(p + 1, del))) {
      p = t;
   }
   if (p) {
      e = p + strlen(del);
      while (*e) {
         *(p++) = *(e++);
      }
      *p = 0;
   }
}

//______________________________________________________________________________
int Cint::Internal::G__templatefunc(G__value* result, const char* funcname, G__param* libp, int hash, int funcmatch)
{
   // Search matching template function, search by name then parameter.
   // If match found, expand template, parse as pre-run and execute it.
   G__Definetemplatefunc* deftmpfunc;
   G__Charlist call_para;
   int store_exec_memberfunc;
   char* pexplicitarg;
   ::Reflex::Scope env_tagnum = G__get_envtagnum();
   ::Reflex::Scope store_friendtagnum = G__friendtagnum;
   G__inheritance* baseclass = 0;
   if (env_tagnum && !env_tagnum.IsTopScope() && !G__struct.baseclass[G__get_tagnum(env_tagnum)]->vec.empty()) {
      baseclass = G__struct.baseclass[G__get_tagnum(env_tagnum)];
   }
   pexplicitarg = (char*) strchr(funcname, '<');
   if (pexplicitarg) {
      // funcname="f<int>" ->  funcname="f" , pexplicitarg="int>"
      int tmp = 0;
      *pexplicitarg = 0;
      if (G__defined_templateclass(funcname)) {
         *pexplicitarg = '<';
         pexplicitarg = 0;
      }
      else {
         ++pexplicitarg;
         G__hash(funcname, hash, tmp);
      }
   }
   call_para.string = 0;
   call_para.next = 0;
   deftmpfunc = &G__definedtemplatefunc;
   // Search matching template function name
   while (deftmpfunc->next) {
      G__freecharlist(&call_para);
      if (
         (deftmpfunc->hash == hash) &&
         !strcmp(deftmpfunc->name, funcname) &&
         (
            pexplicitarg ||
            G__matchtemplatefunc(deftmpfunc, libp, &call_para, funcmatch)
         )
      ) {

         if (
            (deftmpfunc->parent_tagnum != -1) &&
            (env_tagnum != G__Dict::GetDict().GetScope(deftmpfunc->parent_tagnum))
         ) {
            if (baseclass) {
               for (size_t temp = 0; temp < baseclass->vec.size(); ++temp) {
                  if (baseclass->vec[temp].basetagnum == deftmpfunc->parent_tagnum) {
                     goto match_found;
                  }
               }
               // look in global scope (handle for using declaration info
               for (size_t temp = 0; temp < G__globalusingnamespace.vec.size(); ++temp) {
                  if (G__globalusingnamespace.vec[temp].basetagnum == deftmpfunc->parent_tagnum) {
                     goto match_found;
                  }
               }
            }
            deftmpfunc = deftmpfunc->next;
            continue;
         }
         match_found:
         G__friendtagnum = G__Dict::GetDict().GetScope(deftmpfunc->friendtagnum);
         if (pexplicitarg) {
            int npara = 0;
            //858//G__gettemplatearglist(pexplicitarg, &call_para, deftmpfunc->def_para, &npara, -1, 0);
            G__gettemplatearglist(pexplicitarg, &call_para, deftmpfunc->def_para, &npara, -1);
         }
         if (pexplicitarg) {
            int tmp = 0;
            char* p = pexplicitarg - 1;
            pexplicitarg = (char*) malloc(strlen(funcname) + 1);
            strcpy(pexplicitarg, funcname);
            *p = '<';
            G__hash(funcname, hash, tmp);
         }
         else {
            static char cnull[1] = "";
            pexplicitarg = cnull;
         }
         // matches funcname and parameter,
         // then expand the template and parse as prerun
         G__replacetemplate(pexplicitarg, funcname, &call_para, deftmpfunc->def_fp, deftmpfunc->line, deftmpfunc->filenum, &deftmpfunc->def_pos, deftmpfunc->def_para, 0, SHRT_MAX, deftmpfunc->parent_tagnum);
         G__friendtagnum = store_friendtagnum;
         if (pexplicitarg && pexplicitarg[0]) {
            delete pexplicitarg;
         }
         // call the expanded template function
         store_exec_memberfunc = G__exec_memberfunc;
         ::Reflex::Scope scope;
         if (deftmpfunc->parent_tagnum != -1) {
            // Need to do something for member function template
            scope = G__Dict::GetDict().GetScope(deftmpfunc->parent_tagnum);
         }
         else {
            G__exec_memberfunc = 0;
            scope = ::Reflex::Scope::GlobalScope();
         }
         int ret = G__interpret_func(result, funcname, libp, hash, scope, funcmatch, G__TRYNORMAL);
         if (!ret) {
            G__fprinterr(G__serr, "Internal error: template function call %s failed", funcname);
            G__genericerror(0);
            *result = G__null;
         }
         G__exec_memberfunc = store_exec_memberfunc;
         G__freecharlist(&call_para);
         return 1; // match
      }
      deftmpfunc = deftmpfunc->next;
   }
   G__freecharlist(&call_para);
   return 0;  // no match
}

#define G__NOMATCH 0xffffffff

//______________________________________________________________________________
G__funclist* Cint::Internal::G__add_templatefunc(const char* funcnamein, G__param* libp, int hash, G__funclist* funclist, const ::Reflex::Scope p_ifunc, int isrecursive)
{
   // Attempt a function template instantiation by name and argument list, if we succeed, parse as pre-run.
   //
   // Note: Function name is not scoped.
   //
   //fprintf(stderr, "G__add_templatefunc: Begin.  name: %s\n", funcnamein);
   ::Reflex::Scope store_friendtagnum = G__friendtagnum; // Need to restore later.
   // Copy function name for parsing.
   char* funcname = (char*) malloc(strlen(funcnamein) + 1);
   strcpy(funcname, funcnamein);
   // Get scope to begin search at.
   int env_tagnum = G__get_tagnum(p_ifunc);
   //
   //  Check for possible base classes of search starting scope.
   //
   G__inheritance* baseclass = &G__globalusingnamespace;
   if (env_tagnum != -1) {
      baseclass = G__struct.baseclass[env_tagnum];
   }
   if (baseclass->vec.empty()) {
      baseclass = 0;
   }
   //
   //  Find any explicitly specified template arguments.
   //
   char* pexplicitarg = 0;
   char* ptmplt = strchr(funcname, '<');
   if (ptmplt && !strncmp("operator", funcname, ptmplt - funcname)) {
      // We have at least "operator<".
      if (ptmplt[1] == '<') { // We have "operator<<", keep on searching.
         ptmplt = strchr(ptmplt + 2, '<'); // FIXME: Possible array bounds violation.
      }
      else {
         ptmplt = strchr(ptmplt + 1, '<');
      }
   }
   if (ptmplt) {
      if ((env_tagnum != -1) && !strcmp(funcname, G__struct.name[env_tagnum])) { // This is probably a template constructor of a class template.
         ptmplt = 0;
      }
      else {
         *ptmplt = 0;
         G__Definetemplatefunc* p = G__defined_templatefunc(funcname);
         if (p) {
            int tmp = 0;
            G__hash(funcname, hash, tmp);
         }
         else {
            pexplicitarg = ptmplt;
            *ptmplt = '<';
            ptmplt = 0;
         }
      }
   }
   if (pexplicitarg) { // Replace funcname="f<int>", with funcname="f", pexplicitarg="int>"
      *pexplicitarg = 0;
      ++pexplicitarg;
      int tmp = 0;
      G__hash(funcname, hash, tmp);
   }
   //
   //  Search for a matching template function by name.
   //
   G__Charlist call_para;
   call_para.string = 0;
   call_para.next = 0;
   int idx = 0;
   G__Definetemplatefunc* f = &G__definedtemplatefunc;
   for (; f->next; f = f->next, ++idx) {
      G__freecharlist(&call_para);
      //fprintf(stderr, "G__add_templatefunc: idx: %03d hash: %04x name: '%s'\n", idx, f->hash, f->name);
      if (ptmplt) {
         int itmp = 0;
         int ip = 1;
         int c;
         G__StrBuf buf_sb(G__ONELINE);
         char *buf = buf_sb;
         do {
            c = G__getstream_template(ptmplt, &ip, buf, ",>");
            G__checkset_charlist(buf, &call_para, ++itmp, 'u');
         }
         while (c != '>');
      }
      if (
         (f->hash == hash) &&
         !strcmp(f->name, funcname) &&
         (
            G__matchtemplatefunc(f, libp, &call_para, G__PROMOTION) ||
            (pexplicitarg && !libp->paran)
         )
      ) {
         if ((f->parent_tagnum != -1) && (env_tagnum != f->parent_tagnum)) {
            if (!baseclass) {
               continue;
            }
            for (size_t n = 0; n < baseclass->vec.size(); ++n) {
               if (baseclass->vec[n].basetagnum == f->parent_tagnum) {
                  goto match_found;
               }
            }
            continue;
         }
         match_found:
         G__friendtagnum = G__Dict::GetDict().GetScope(f->friendtagnum);
         if (pexplicitarg) {
            int npara = 0;
            //858//G__gettemplatearglist(pexplicitarg, &call_para, f->def_para, &npara, -1, 0);
            G__gettemplatearglist(pexplicitarg, &call_para, f->def_para, &npara, -1);
         }
         if (pexplicitarg) {
            int tmp = 0;
            G__hash(funcname, hash, tmp);
         }
         // It matches funcname and parameter, now expand the template and parse as prerun.
         //fprintf(stderr, "G__add_templatefunc:  Calling  G__replacetemplate.  funcnamein: '%s'  funcname: '%s'\n", funcnamein, funcname);
         G__replacetemplate(funcname, funcnamein, &call_para, f->def_fp, f->line, f->filenum, &f->def_pos, f->def_para, 0, SHRT_MAX /* large enough number */, f->parent_tagnum);
         //fprintf(stderr, "G__add_templatefunc:  Finished G__replacetemplate.  funcnamein: '%s'  funcname: '%s'\n", funcnamein, funcname);
         G__freecharlist(&call_para);
         G__friendtagnum = store_friendtagnum;
         //
         //  Search for the instantiated template function.
         //
         int index = p_ifunc.FunctionMemberSize() - 1;
         ::Reflex::Member func = p_ifunc.FunctionMemberAt(index);
         if (func && !strcmp(funcnamein, func.Name().c_str())) {
            if (!G__get_funcproperties(func)->entry.p && (G__globalcomp == G__NOLINK)) {
               continue; // Only a prototype, continue searchng for definition.
            }
            funclist = G__funclist_add(funclist, func, index, 0);
            if (
               ((int) func.FunctionParameterSize() < libp->paran) ||
               (
                  ((int) func.FunctionParameterSize() > libp->paran) &&
                  !func.FunctionParameterSize(true)
               )
            ) {
               funclist->rate = G__NOMATCH;
            }
            else {
               G__rate_parameter_match(libp, func, funclist, isrecursive);
            }
         }
      }
   }
   G__freecharlist(&call_para);
   if (funcname) {
      free(funcname);
   }
   //fprintf(stderr, "G__add_templatefunc: End.  name: '%s'\n", funcnamein);
   return funclist;
}

#undef G__NOMATCH

//______________________________________________________________________________
char* Cint::Internal::G__gettemplatearg(int n, G__Templatearg* def_para)
{
   // search matches for template argument
   G__ASSERT(def_para);
   for (int i = 1; i < n; ++i) {
      if (def_para->next) {
         def_para = def_para->next;
      }
   }
   return def_para->string;
}

//______________________________________________________________________________
extern "C" G__Definedtemplateclass* G__defined_templateclass(const char* name)
{
   // Check if the template class is declared
   // but maybe in future I might need this to handle case 4,5
   G__Definedtemplateclass* deftmplt;
   int hash;
   int temp;
   char* dmy_struct_offset = 0;
   G__StrBuf atom_name_sb(G__LONGLINE);
   char* atom_name = atom_name_sb;
   ::Reflex::Scope env_tagnum = G__get_envtagnum();
   ::Reflex::Scope scope_tagnum;
   G__inheritance* baseclass = 0;
   // return if no name
   if (
      !name[0] ||
      strchr(name, '.') ||
      strchr(name, '-') ||
      strchr(name, '(') ||
      isdigit(name[0]) ||
      (
         !isalpha(name[0]) &&
         (name[0] != '_') &&
         (name[0] != ':')
      )
   ) {
      return 0;
   }
   // get a handle for using declaration info
   if (env_tagnum && !env_tagnum.IsTopScope()) {
      int tagnum = G__get_tagnum(env_tagnum);
      assert(tagnum != -1);
      if (!G__struct.baseclass[tagnum]->vec.empty()) {
         baseclass = G__struct.baseclass[tagnum];
      }
   }
   // scope operator resolution, A::templatename<int> ...
   strcpy(atom_name, name);
   G__hash(atom_name, hash, temp)
   int intScopeTagnum = -1;
#ifdef __GNUC__
#else // __GNUC__
#pragma message(FIXME("Replace G__Scopeoperator's int& arg by Scope& if needed at all"))
#endif // __GNUC__
   int scope = G__scopeoperator(atom_name, &hash, &dmy_struct_offset, &intScopeTagnum);
   scope_tagnum = G__Dict::GetDict().GetScope(intScopeTagnum);
   // Don't crash on a null name (like 'std::').
   if (!atom_name[0]) {
      return 0;
   }
   // search for template name and scope match
   deftmplt = &G__definedtemplateclass;
   G__Definedtemplateclass* candidate = 0;
   for (deftmplt = &G__definedtemplateclass; deftmplt->next; deftmplt = deftmplt->next) {
      if ((hash == deftmplt->hash) && !strcmp(atom_name, deftmplt->name)) {
         if (scope != G__NOSCOPEOPR) {
            // look for ordinary scope resolution
            if (
               (
                  (!scope_tagnum || scope_tagnum.IsTopScope()) &&
                  (
                     (deftmplt->parent_tagnum == -1) ||
                     (env_tagnum == G__Dict::GetDict().GetScope(deftmplt->parent_tagnum))
                  )
               ) ||
               (scope_tagnum == G__Dict::GetDict().GetScope(deftmplt->parent_tagnum))
            ) {
               return deftmplt;
            }
         }
         else if (env_tagnum == G__Dict::GetDict().GetScope(deftmplt->parent_tagnum)) {
            // Exact environment scope match
            return deftmplt;
         }
         else if (!scope_tagnum || scope_tagnum.IsTopScope()) {
            ::Reflex::Scope env_parent_tagnum = env_tagnum;
            if (baseclass && !candidate) {
               // look for using directive scope resolution
               for (size_t temp1 = 0; temp1 < baseclass->vec.size(); ++temp1) {
                  if (baseclass->vec[temp1].basetagnum == deftmplt->parent_tagnum) {
                     candidate = deftmplt;
                  }
               }
            }
            // look for enclosing scope resolution
            while (!candidate && env_parent_tagnum && !env_parent_tagnum.IsTopScope()) {
               env_parent_tagnum = env_parent_tagnum.DeclaringScope();
               if (env_parent_tagnum == G__Dict::GetDict().GetScope(deftmplt->parent_tagnum)) {
                  candidate = deftmplt;
                  break;
               }
               int parent_tagnum = G__get_tagnum(env_parent_tagnum);
               if (parent_tagnum != -1) {
                  if (G__struct.baseclass[parent_tagnum]) {
                     for (size_t temp1 = 0; temp1 < G__struct.baseclass[parent_tagnum]->vec.size(); ++temp1) {
                        if (G__struct.baseclass[parent_tagnum]->vec[temp1].basetagnum == deftmplt->parent_tagnum) {
                           candidate = deftmplt;
                           break;
                        }
                     }
                     if (candidate) {
                        break;
                     }
                  }
               }
            }
            // look in global scope (handle for using declaration info
            if (!candidate) {
               for (size_t temp1 = 0; temp1 < G__globalusingnamespace.vec.size(); ++temp1) {
                  if (G__globalusingnamespace.vec[temp1].basetagnum == deftmplt->parent_tagnum) {
                     candidate = deftmplt;
                  }
               }
            }
         }
      }
   }
   return candidate;
}

//______________________________________________________________________________
G__Definetemplatefunc* Cint::Internal::G__defined_templatefunc(const char* name)
{
   // Check if the template function is declared
   G__Definetemplatefunc* deftmplt;
   int hash;
   int temp;
   char* dmy_struct_offset = 0;
   G__StrBuf atom_name_sb(G__LONGLINE);
   char* atom_name = atom_name_sb;
   ::Reflex::Scope env_tagnum = G__get_envtagnum();
   ::Reflex::Scope scope_tagnum;
   G__inheritance* baseclass;
   // return if no name
   if (!name[0] || strchr(name, '.') || strchr(name, '-') || strchr(name, '(')) {
      return 0;
   }
   // get a handle for using declaration info
   baseclass = 0;
   if (!env_tagnum.IsTopScope() && !G__struct.baseclass[G__get_tagnum(env_tagnum)]->vec.empty()) {
      baseclass = G__struct.baseclass[G__get_tagnum(env_tagnum)];
   }
   // scope operator resolution, A::templatename<int> ...
   strcpy(atom_name, name);
   G__hash(atom_name, hash, temp)
   int intScopeTagnum = G__get_tagnum(scope_tagnum);
   G__scopeoperator(atom_name, &hash, &dmy_struct_offset, &intScopeTagnum);
   scope_tagnum = G__Dict::GetDict().GetScope(intScopeTagnum);
#ifdef __GNUC__
#else // __GNUC__
#pragma message(FIXME("G__scopeoperator should take a Reflex::Type if we need it at all!"))
#endif // __GNUC__
   // Don't crash on a null name (like 'std::').
   if (!atom_name[0]) {
      return 0;
   }
#ifdef __GNUC__
#else // __GNUC__
#pragma message(FIXME("Use proper Reflex func lookup here"))
#endif // __GNUC__
   // search for template name and scope match
   deftmplt = &G__definedtemplatefunc;
   while (deftmplt->next) {
      if ((hash == deftmplt->hash) && !strcmp(atom_name, deftmplt->name)) {
         // look for ordinary scope resolution
#ifdef __GNUC__
#else // __GNUC__
#pragma message (FIXME("This part also took using statements of base classes and enclosing scopes into account, we don't! Fix by using Reflex func lookup."))
#endif // __GNUC__
         Reflex::Scope declScope = G__Dict::GetDict().GetScope(deftmplt->parent_tagnum);
         if (
            (
               (!scope_tagnum || scope_tagnum.IsTopScope()) &&
               (!G__tagdefining || G__tagdefining.IsTopScope()) &&
               (
                  (deftmplt->parent_tagnum == -1) ||
                  (env_tagnum == declScope)
               )
            ) ||
            (
               (scope_tagnum == declScope) &&
               (
                  (!G__tagdefining || G__tagdefining.IsTopScope()) ||
                  (G__tagdefining == declScope)
               )
            ) ||
            std::find(env_tagnum.UsingDirective_Begin(), env_tagnum.UsingDirective_End(), declScope) != env_tagnum.UsingDirective_End()
         ) {
            return(deftmplt);
         }
         else if (!scope_tagnum || scope_tagnum.IsTopScope()) {
            ::Reflex::Scope env_parent_tagnum = env_tagnum;
            if (baseclass) {
               // look for using directive scope resolution
               for (size_t temp2 = 0; temp2 < baseclass->vec.size(); ++temp2) {
                  if (baseclass->vec[temp2].basetagnum == deftmplt->parent_tagnum) {
                     return deftmplt;
                  }
               }
            }
            // look for enclosing scope resolution
            while (env_parent_tagnum && !env_parent_tagnum.IsTopScope()) {
               env_parent_tagnum = env_parent_tagnum.DeclaringScope();
               if (
                  (env_parent_tagnum == G__Dict::GetDict().GetScope(deftmplt->parent_tagnum)) &&
                  (
                     (!G__tagdefining || G__tagdefining.IsTopScope()) ||
                     (G__tagdefining == G__Dict::GetDict().GetScope(deftmplt->parent_tagnum))
                  )
               ) {
                  return deftmplt;
               }
               if (G__struct.baseclass[G__get_tagnum(env_parent_tagnum)]) {
                  size_t nbases = G__struct.baseclass[G__get_tagnum(env_parent_tagnum)]->vec.size();
                  for (size_t temp2 = 0; temp2 < nbases; ++temp2) {
                     if (G__struct.baseclass[G__get_tagnum(env_parent_tagnum)]->vec[temp2].basetagnum == deftmplt->parent_tagnum) {
                        return deftmplt;
                     }
                  }
               }
            }
            // look in global scope (handle for using declaration info
            for (size_t temp1 = 0; temp1 < G__globalusingnamespace.vec.size(); ++temp1) {
               if (G__globalusingnamespace.vec[temp1].basetagnum == deftmplt->parent_tagnum) {
                  return deftmplt;
               }
            }
         }
      }
      deftmplt = deftmplt->next;
   }
   return 0;
}

//______________________________________________________________________________
G__Definetemplatefunc* Cint::Internal::G__defined_templatememfunc(const char* name)
{
   //
   // t.Handle<int>();
   // a.t.Handle<int>();
   // a.f().Handle<int>();
   //
   char* p;
   char* p1;
   char* p2;
   G__StrBuf atom_name_sb(G__LONGLINE);
   char* atom_name = atom_name_sb;
   int store_asm_noverflow = G__asm_noverflow;
   G__Definetemplatefunc* result = 0;
   // separate "t" and "Handle"
   strcpy(atom_name, name);
   p1 = strrchr(atom_name, '.');
   p2 = G__strrstr(atom_name, "->");
   if (!p1 && !p2) {
      return result;
   }
   if ((p1 > p2) || !p2) {
      *p1 = 0;
      p = p1 + 1;
   }
   else {
      *p2 = 0;
      p = p2 + 2;
   }
   // "t" as name "Handle" as p
   G__suspendbytecode();
   {
      ::Reflex::Scope tagnum = G__getobjecttagnum(atom_name);
      if (tagnum) {
         ::Reflex::Scope store_def_tagnum = G__def_tagnum;
         ::Reflex::Scope store_tagdefining = G__tagdefining;
         // Have to look at base class.
         G__def_tagnum = tagnum;
         G__tagdefining = tagnum;
         result = G__defined_templatefunc(p);
         G__def_tagnum = store_def_tagnum;
         G__tagdefining = store_tagdefining;
         if (!result) {
            int len = strlen(p);
            p[len++] = '<';
            p[len] = 0;
            for (::Reflex::Member_Iterator f = tagnum.FunctionMember_Begin(); f != tagnum.FunctionMember_End(); ++f) {
               if (!strncmp(f->Name().c_str(), p, len)) {
                  result = (G__Definetemplatefunc*) G__PVOID;
               }
            }
            p[len-1] = 0;
         }
      }
   }
   G__asm_noverflow = store_asm_noverflow;
   if (p1 && !(*p1)) {
      *p1 = '.';
   }
   if (p2 && !(*p2)) {
      *p2 = '-';
   }
   return result;
}

//______________________________________________________________________________
static ::Reflex::Type G__getobjecttagnum(char *name)
{
   ::Reflex::Type result;
   char* p;
   char* p1 = strrchr(name, '.');
   char* p2 = G__strrstr(name, "->");
   if (!p1 && !p2) {
      int ig15;
      int itmpx;
      int varhash;
      char* store_struct_offset1 = 0;
      char* store_struct_offset2 = 0;
      G__hash(name, varhash, itmpx);
      ::Reflex::Member var = G__find_variable(name, varhash, G__p_local, ::Reflex::Scope::GlobalScope(), &store_struct_offset1, &store_struct_offset2, &ig15, 0);
      if (var && (tolower(G__get_type(var.TypeOf())) == 'u') && (G__get_tagnum(var.TypeOf()) != -1)) {
         result = var.TypeOf();
         return result;
      }
      else {
         char* p3 = strchr(name, '(');
         if (p3) {
            // LOOK FOR A FUNCTION
         }
      }
   }
   else {
      if ((p1 > p2) || !p2) {
         *p1 = 0;
         p = p1 + 1;
      }
      else {
         *p2 = 0;
         p = p2 + 2;
      }

      result = G__getobjecttagnum(name);
      if (result) {
         // TO BE IMPLEMENTED
         // G__var_array* var = G__struct.memvar[result];
         // G__ifunc_table* ifunc = G__struct.memfunc[result];
      }
   }
   if (p1 && !(*p1)) {
      *p1 = '.';
   }
   if (p2 && !(*p2)) {
      *p2 = '-';
   }
   return result;
}

