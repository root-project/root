/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file symbols.cxx
 * Leandro Franco
 ************************************************************************
 * Description:
 *  This file performs the reading and registering of library symbols.
 *  This is done creating the dictionaries and the idea is to add a new 
 *  field to the memfunc setup containing the mangled name for the symbol
 ************************************************************************
 * Copyright(c) 1995~2004  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"
#include "Api.h"

#include <cxxabi.h>
#include <dlfcn.h>
#include <iostream>
#include <fstream>

#ifndef __APPLE__  /* Apple MacOS X */
#include <link.h> // lib stuff
#else
#include <libgen.h> // needed for basename
#endif

#include <vector>
#include <list>
#include <set>
#include <string>
#include <map>
#include <algorithm>
using namespace std;

static int gDebug = 0;

inline unsigned long hash(const char* str)
{
  unsigned long hash = 5381;

  while(*str) {
    hash *= 5;
    hash += *(str++);
  }
  return hash;
}

struct ptr_delete {
  template< class T >
  void operator()( T * const p ) const {
    delete p;
  }
};

// This is a simple class that contains the information
// needed for a symbol. In particular, its name and address
class TSymbol {
private:
  TSymbol(const TSymbol& sym);       //Avoid copying ptrs
  TSymbol& operator=(const TSymbol& sym);  //Avoid copying ptrs

  static const short CONST  = 0x01; // first bit 
  static const short DEST   = 0x02; // second bit 

public:
  unsigned int   fClassHash;

  // This points directly to the name in the library
  char*    fMangled;
      
  short fIdSpecial; // Is it a constructor or destructor?

  TSymbol()
    : fClassHash(0), fMangled(0), fIdSpecial(0)
  {
  }

  TSymbol(const char* mangled)
    : fClassHash(0)
  {
    fMangled = new char[strlen(mangled) + 1];
    strcpy(fMangled, mangled);

    fIdSpecial = 0;
  }

  ~TSymbol()
  {
    delete [] fMangled;
  }

  void SetHash(char *demanstr)
  {
    if(!demanstr)
      return;

    string demangled(demanstr);

    // First we have to tell it that we are only interested for
    // the strings that are _before_ the ()
    string::size_type ind =  demangled.find("(");

    // maybe this symbol is not a function
    if (ind == string::npos)
      return;

    string::size_type start = 0;
    string::size_type end   = demangled.find(" ", start);
      
    while(start != string::npos) {
      if ((end == string::npos) || (end > ind))
        end = ind;
         
      int ncolon=0;
      string::size_type cstart = start;
      string::size_type cidx = start;
      while((cidx != string::npos) && (cidx < ind)){
        cidx = demangled.find("::", cstart);

        if((cidx != string::npos) && (cidx < ind)){
          ++ncolon;
          cstart = cidx+2;
        }
      }
      
      // 15-10-07
      // Be careful with namespaces...
      // things like TBits::TReference::~TReference()
      // will confuse our naive algorithm... instead of just looking
      // for '::', look for the last pair of '::'
      string::size_type icolon = string::npos;
      string::size_type istart = start;
      for (int i=0;i<ncolon;i++){
        if(icolon != string::npos)
          start = istart;

        icolon = demangled.find("::", istart);
        istart = icolon+2;
      }
      //string::size_type icolon = demangled.find("::", start);
         
      if (icolon != string::npos && icolon < ind){
        // Now hash the class name also
        // (it will make things easier when we try to register
        // symbols by class)
        //string classname = string(demangled, start, icolon);
        string classname = demangled.substr(0, icolon);
        string classname_noname = demangled.substr(start, icolon-start);
        string protoname = demangled.substr(icolon+2, ind - (icolon+2));


        // 11-10-07
        // Get rid of the "<>" part in something like TParameter<float>::TParameter()
        string::size_type itri = classname.find("<");

        string classname_notemp;
        if(itri != string::npos){
          classname_notemp = classname_noname.substr(0, itri);
        }
        else
          classname_notemp = classname_noname;

        // ** constructors
        if ( classname_notemp == protoname) {
          SetConst();
          // if this not the constructor in charge then just continue
          if(!(strstr(fMangled, classname_notemp.c_str()) && strstr(fMangled, "C1")  )) {
            // This is not the constructor in-charge... ignore it
            return;
          }
        }

        // ** destructors
        string dest("~");
        dest += classname_notemp;
        if ( dest == protoname){
          SetDest();
               
          if(strstr(fMangled, classname_notemp.c_str()) && strstr(fMangled, "D0")) {
            // This is the deleting constructor

            // 27-07-07
            // We ignore the deleting constructor since we do
            // that job in newlink
            return;
          }
          else if( strstr(fMangled, classname_notemp.c_str()) && strstr(fMangled, "D1") ){
            // This is the in-charge (non deleting) constructor
            // Dont do anything but keep it
          }
          else {
            // Ignoring destructor not-in-charge for" << classstr.Data() << endl; 
            return;
          }
        }
        // 05-11-07
        // We can get something like:
        // ROOT::Math::SVector<double, 2u>
        // When we have actually declared
        // ROOT::Math::SVector<double,2>
        // in ROOT... so the hashes wont match..
        // try to convert the former in the latter.
        
        string::size_type pos_smaller  = classname.find_first_of("<", 0);
        string::size_type pos_greater  = classname.find_last_of(">", classname.size()-1);

        string::size_type index=0;
        while(index < classname.size()){
          if(classname[index]==' ' && 
             (index>0 && !isalpha(classname[index-1])) && 
             (index<classname.size()-1 && !isalpha(classname[index+1])) ){
            classname.erase(index,1);
            pos_greater  = classname.find_last_of(">", classname.size()-1);
          }
          else if( (pos_smaller != string::npos) &&
                   (pos_greater != string::npos) &&
                   index>pos_smaller && index<pos_greater ) {
            // How is the isinteger(char) when using the stl?
            if( isdigit(classname[index]) && classname[index+1]=='u') {
              classname.erase(index+1,1);
              pos_greater  = classname.find_last_of(">", classname.size()-1);
            }
            else
              index++;
          }
          else
            index++;
        }
        fClassHash = hash(classname.c_str());
        return;
      }
      else {
        fClassHash = hash(demanstr);
        //this->SetName(demangled.Data());
        return;
      }
      start = demangled.find(" ", start);
      if(start != string::npos)
        end = demangled.find(" ", start);
    }
  }

  unsigned long Hash() const
  {
    // The hash of a symbol is going to be hash of the method's name
    // ie. "TObject::Dump" for the method dump in TObject
    return fClassHash;
  }

  char* Demangle()
  {
    // demangles the string pointed by mangled and returns 
    // a new char* containing the demangled name
    // rem to free that value

    // 22/05/07
    // Instead of copying the fMangled name just do the demangling
    // I dont know why the symbols start with __Z if the ABI wants them
    // to start with _Z... until understanding it just hack your way through
    const char *cline = fMangled;
    if ( cline[0]=='_' && cline[1]=='_') {
      cline = cline++;
    }
      
    // We have the symbols string in "line"
    // now try to demangle it
    int status = 0;
    char *name = abi::__cxa_demangle(cline, 0, 0, &status);
    if (!name)
    {
      // It probably wasnt a valid symbol...
      // to be sure check the status code
      // "error code = 0: success";
      // "error code = -1: memory allocation failure";
      // "error code = -2: invalid fMangled name";
      // "error code = -3: invalid arguments";
      return 0;
    }
      
    // Ignore this symbols if it was generated by cint
    if( !strncmp( name, "G__", 3) || strstr(name, "thunk")){
      if (name)
        free(name);
         
      return 0;
    }
      
    return name;
  }
   
  bool IsConst()
  {
    return fIdSpecial & CONST;
  }

  bool IsDest()
  {
    return fIdSpecial & DEST;
  }

  void SetConst()
  {
    fIdSpecial &= CONST;
  }

  void SetDest()
  {
    fIdSpecial &= DEST;
  }

  //ClassDef(TSymbol,0)
};


// This will be the lookup for the symbols. The idea is to have
// a list of the available (open) libraries with links to 
// its symbols and the classes that have already been registered
class TSymbolLookup {

private:
  TSymbolLookup(const TSymbolLookup& sym);             //Avoid copying ptrs
  TSymbolLookup& operator=(const TSymbolLookup& sym);  //Avoid copying ptrs

public:
  const char*    fLibname; // can this be just a reference? wont it change in CInt?
  std::list<std::string>  fRegistered;
  std::list<TSymbol*>    *fSymbols;

  TSymbolLookup()
  {
    fLibname    = 0;
    fSymbols    = 0;
  }

  TSymbolLookup(const char *libname, std::list<TSymbol*> *symbols)
  {
    fLibname    = libname;
    fSymbols    = symbols;
  }

  ~TSymbolLookup()
  {
    std::for_each( fSymbols->begin(), fSymbols->end(), ptr_delete() );
    fSymbols->clear();
  }
   
  void AddRegistered(const string &classname)
  {
    fRegistered.push_back(classname);
  }

  unsigned long Hash() const
  {
    // This is just the name of the library
    // (always remember that this is the fullpath)

    if (fLibname)
      return hash(fLibname);

    return 0;
  }

  bool IsRegistered(string &classname)
  {
    // Let's check if this class has already been registered
    std::list<string>::iterator iter = find(fRegistered.begin(), fRegistered.end(), classname);
    if (iter != fRegistered.end())
      return true;

    return false;
  }
   
  void SetLibname(const char* libname)
  {
    // Give a libname to this object
    //delete fLibname;

    fLibname = libname;
  }

  void SetSymbols(std::list<TSymbol*> *table)
  {
    // Associate a symbol table to this library
    std::for_each( fSymbols->begin(), fSymbols->end(), ptr_delete() );
    fSymbols->clear();

    fSymbols = table;
  }

};


std::multimap<std::string, TSymbolLookup*>*
G__get_cache()
{
  // sort of singleton for the symbol cache
  static multimap<string, TSymbolLookup*> G__cache;
  return &G__cache;
}


//______________________________________________________________________________
//______________________________________________________________________________
// Methods used to register the function pointer (or method pointer)
// of a given signature inside CINT. This should probably go to another file
//
//______________________________________________________________________________
//______________________________________________________________________________

//______________________________________________________________________________
void MapDependantTypesX()
{
  // When we demangled the symbols at registering time, we might find certain 
  // strange types. This is because CInt doesn't know them as such and
  // gcc/abi reports them in a different way.
  // This is not needed anymore since with the current algorithm we will just
  // ignore a function with that type (generating the stub).
  // This function should probably be erased. It's not good to hard code
  // defines.

  int more = 0;
  G__FastAllocString prompt(G__ONELINE);

  // Like MapDependantTypes but for the class TGX11

  G__process_cmd("typedef struct _XDisplay  { } Display;", prompt, &more,0, 0);
  G__process_cmd("typedef struct _XGC       { } *GC;", prompt, &more,0, 0);
  G__process_cmd("typedef struct _XEvent    { } XEvent", prompt, &more,0, 0);
  G__process_cmd("typedef struct _XImage    { } Image", prompt, &more,0, 0);
  G__process_cmd("typedef struct            { } XFontStruct;", prompt, &more,0, 0);
  G__process_cmd("typedef struct FT_Bitmap_ { } FT_Bitmap", prompt, &more,0, 0);
}

//______________________________________________________________________________
static std::list<TSymbol*>*
G__symbol_list(const char* lib)
{
  // 05-07-07
  // This is the second attempt so instead of getting the symbols from the 
  // library (in memory) do it from a file called (libCore.so -> libCore.nm),
  // where such path is passed as a parameter here.
  // That file only constain a list of symbols (mangled)

  // I have to use a new HashList here since the hash changes...
  // before it was zero and now it's teh real hash... it has
  // to be modified for something more practical!!!
  std::list<TSymbol*> *symbol_list = new std::list<TSymbol*>;
  
  FILE * fp;
  char *line = NULL;
  size_t len_line = 0;
  ifstream myfile (lib);

  fp = fopen(lib, "r");
  if (fp) {
    while (getline(&line, &len_line, fp) != -1) {
      // The stupid getline will add the eol character if it's found
      size_t len = strlen(line);
      if(len>1 && line[len-1]=='\n') {
        line[len-1] = '\0';
      }

      TSymbol *symbol = new TSymbol((const char*)line);
      char* demangled = symbol->Demangle();
      symbol->SetHash(demangled);
      
      // If this symbol has no shape... dont put it inside the list
      if (symbol->Hash()) {
        symbol_list->push_back(symbol);
            
        if(gDebug>2) {
          cerr << " --- Keeping symbol: " << demangled << endl;
          cerr << " --- hash          : " << symbol->Hash() << endl;
        }
      }
      else
        delete symbol;
      
      // Dont forget to delete "name"
      // rem that __cxa_demangle will allocate the space needed for this
      if (demangled)
        free(demangled);
    }
    fclose(fp);
  }
  else 
    cout << "Error: couldnt open file: " << lib << endl;
  

  if (line)
    free(line);

  return symbol_list;
}


//______________________________________________________________________________
int G__register_pointer(const char *classname, const char *method, const char *proto,
                        const char *mangled_name, int isconst)
{
  // Try to register the function pointer 'ptr' for a given method
  // of a given class with a certain prototype, i.e. "char*,int,float".
  // If the class is 0 the global function list will be searched (not yet).
  // Returns -1 if there was a problem and 0 if it was succesful

  struct G__ifunc_table_internal *ifunc;
  char *funcname;
  char *param;
  long index;
  long offset;
  G__ifunc_table* iref = 0;
   
  int tagnum = -1;
  ifunc = 0;

  if(strcmp(classname, "")!=0) {
    // 05-11-07
    // Be careful!!!!
    // G__defined_tagname can change the value of classname... why??? 
    // so something like:
    // "ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepStd<double,2,2>>"
    // can be translated to something like 
    // "ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepStd<double,2,2> >"
    // Which might look the same but can give unexpected results (imagine 
    // you truncate the '>' in the second case... )
    // this is very ugly...
    tagnum = G__defined_tagname(classname, 2);
    ifunc = G__struct.memfunc[tagnum];
  }

  // This means it doesn't belog to a class
  // i.e. free standing
  if (!classname || strcmp(classname, "")==0) {
    /* Search for method */
    funcname = (char*)method;
    param = (char*)proto;
    //iref = G__get_methodhandle(funcname,param,G__p_ifunc,&index,&offset,0,0,2,isconst?G__CONSTFUNC:0);
    iref = G__get_methodhandle_noerror(funcname,param,G__get_ifunc_ref(G__p_ifunc),&index,&offset,0,0,2,isconst?2:1);
  }
  else {
    /* Search for method */
    funcname = (char*)method;
    param = (char*)proto;
    //iref = G__get_methodhandle(funcname,param,ifunc,&index,&offset,0,0,2,isconst?G__CONSTFUNC:0);
    iref = G__get_methodhandle_noerror(funcname,param,G__get_ifunc_ref(ifunc),&index,&offset,0,0,2,isconst?2:1);
  }
  if (!iref)
    return -1;

  ifunc = G__get_ifunc_internal(iref);
  // set pointer to interface method and arguments
  // Don't do it if the name has already been written
  if(!ifunc->mangled_name[index])
    G__savestring(&ifunc->mangled_name[index],(char*)mangled_name);

  return 0;
}

//______________________________________________________________________________
void G__register_class(const char *libname, const char *clstr)
{
  // It tries to register the address of all the
  // symbols in a library.
  //
  // For this it will:
  //
  // * Load the library
  // * Traverse it and read the symbols with their addresses
  // * Demangle the symbols
  // * Try to register each pair demangled symbol<->address
  //
  // Note: we dont really use the complete demangled string
  // only its signature
  //
  // If clstr (a string with a class) is specified then we will
  // try to register only the methods belonging to that class,
  // if it's Null we will try to register all the free standing
  // functions of that library
   
  string classname;
  if(clstr)
    classname = clstr;

   
  // Take out spaces from template declarations
  string::size_type p_smaller  = classname.find_first_of("<", 0);
  string::size_type p_greater  = classname.find_last_of(">", classname.size()-1);

  if( (p_smaller != string::npos) &&
      (p_greater != string::npos)){

    // ** Pre-process a token
    string::size_type index=0;
    while(index < classname.size()){
      if( index>p_smaller && index<p_greater ) {
        if(classname[index]==' ' &&
           (index>0 && !isalpha(classname[index-1])) && 
           (index<classname.size()-1 && !isalpha(classname[index+1]))){
          classname.erase(index,1);
          p_greater  = classname.find_last_of(">", classname.size()-1);
        }
        // How is the isinteger(char) when using the stl?
        else if( (classname[index]=='0' || classname[index]=='1' || classname[index]=='2' || classname[index]=='3' || classname[index]=='4' ||
                  classname[index]=='5' || classname[index]=='6' || classname[index]=='7' || classname[index]=='8' || classname[index]=='9')
                 && classname[index+1]=='u'){
          classname.erase(index+1,1);
          p_greater  = classname.find_last_of(">", classname.size()-1);
        }
        else
          index++;
      }
      else
        index++;
    }
    // ** Pre-process a token
  }

  unsigned int  classhash = hash(classname.c_str());
  int nreg = 0;
  std::list<TSymbol*> *demangled = 0;
  int nerrors = 0;
  int isFreeFunc = classname.empty();

  if(!libname || strcmp(libname,"")==0 ){
    if (gDebug > 0) {
      cerr << "****************************************" << endl;
      cerr << "Warning: Null library for class " << clstr << endl;
      cerr << "****************************************" << endl;
    }
    return;
  }
    
  // When we execute something from a macro, Register can be called with a libname
  // something.C so let's check and dont do anything if this not a library
  int len = strlen(libname);
  if (strlen(libname) > 2 && !strcmp(libname+len-2, ".h"))
    return;

  if (strlen(libname) > 3 && !(!strcmp(libname+len-3, ".sl") ||
                               !strcmp(libname+len-3, ".dl") ||
                               !strcmp(libname+len-4, ".dll")||
                               !strcmp(libname+len-4, ".DLL")||
                               !strcmp(libname+len-3, ".so") ||
                               !strcmp(libname+len-3, ".nm") ||
                               !strcmp(libname+len-2, ".a"))) {
    if (gDebug > 0) {
      cerr << "****************************************" << endl;
      cerr << "Error: " << libname << " doesnt look like a valid library" << endl;
      cerr << "****************************************" << endl;
    }
    return;
  }

  if (gDebug > 0) {
    cerr << "****************************************" << endl;
    cerr << "Reading symbols from library:" << libname << endl;
  }


  /*********************************/
  // small hack to avoid funny types
  char *basec, *bname;
  basec = strdup(libname);
  bname = basename(basec);

  if(strcmp(bname, "libGX11.so")==0) {
    MapDependantTypesX();
  }
  free(basec);
  /*********************************/
   
   
  // 09/05/07
  // Here we have the first try to implement the symbol cache.
  // The first thing to do is to look for this library in fSymbolTable
  // Note: Remember to keep all the libraries with their full path
  multimap<string,TSymbolLookup*>           *symbolt = G__get_cache();
  multimap<string,TSymbolLookup*>::iterator  iter    = symbolt->find(string(libname));
  TSymbolLookup *symt = 0;
  if (iter != symbolt->end()) {
    symt = (TSymbolLookup *) (iter->second);
  }

  if (gDebug > 0)
    cerr << "+++ Looking for the symbol table of: " << libname << endl;

  // We found it... now check is this class was already registered
  if(symt) {
    if (gDebug > 0)
      cerr << "+++ The library has already been read " << endl << endl;

    if (gDebug > 0)
      cerr << "+++ Checking if the class '" << classname << "' has been registered" << endl;
    // Rem that clstr can be NULL when registering globals.
    // So let's convert that NULL to ""
    if(symt->IsRegistered(classname)) {
      if (gDebug > 0)
        cerr << "+++ class '" << classname << "' has been registered" << endl << endl;

      return; // Yahoo... we dont have to do anything
    }
    else{
      // This means the library is there with all the symbols but we still
      // have to do the registering
      if (gDebug > 0)
        cerr << "+++ class '" << classname << "' has NOT been registered" << endl << endl;
      demangled = symt->fSymbols;

      // Assume this class has been registered (is what we do next
      if (gDebug > 0)
        cerr << "+++ registering symbols for class '" << classname << "'" << endl << endl;
      symt->AddRegistered(classname);
    }
      
  }
  else{
    if (gDebug > 0)
      cerr << "+++ This is a new library " << endl << endl;

    demangled = G__symbol_list(libname);

    if (!demangled) {
      cerr << "Error reading/demangling the symbols" << endl;

      return;
    }

    if (gDebug > 0)
      cerr << "+++ Add library '" << libname << "' to the list of libraries" << endl << endl;

    // And now that we have the symbols here we add them to the cache (for future lookups)
    symt = new TSymbolLookup(libname, demangled);
    G__get_cache()->insert(make_pair(string(libname), symt));

    // Assume this class has been registered (is what we do next)
    if (gDebug > 0)
      cerr << "+++ registering symbols for class '" << classname << "'" << endl << endl;
    symt->AddRegistered(classname);
  }

  // To register a new address we need:
  //
  // -- The class, asgiven by an object: TClass *cl
  // -- The mthod.. as a string        : const char *method
  // -- the proto.. as another string  : const char *proto
  // -- the address of the pointer     : void *ptr)
  // Now we have have the Hash table with the demangled symbols
  // and the addresses
  //
  // And that's for each entry in the hashing table

  if (gDebug > 0) {
    cerr << "****************************************" << endl;
    cerr << "Registering symbols " << endl;
  }

  std::list<TSymbol*>::iterator list_iter = demangled->begin();
  while( list_iter != demangled->end()) {
    TSymbol *symbol = (TSymbol*) *list_iter;

    if( !isFreeFunc && (classhash != symbol->fClassHash) ){
      ++list_iter;
      continue;
    }

    char* deman = symbol->Demangle();
    string sig(deman);
      
    if(deman){
      free(deman);
      deman = 0;
    }
    string classstr = "";
    string classstr_noname = "";
    string protostr = "";

    // 16/04/2007
    // We cant tokenize with () only because that spoils things
    // when we have the "operator()" like in:
    //  TMatrixT<double>::operator()(int, int)
    // This is extreamly annoying and proves that this whole
    // parsing section should be rewritten following a set of rules
    // instead of the piñata paradigm

    int ncolon=0;
    string::size_type start = 0;
    string::size_type cstart = start;
    string::size_type cidx = start;
    string::size_type ind=sig.find("(");

    if(sig.find("operator()")!=string::npos)
      ind=sig.find("(", ind+1);

    // Another ugly hack... rewrite this to handle things like:
    // (anonymous namespace)::CreateIntRefConverter(long)
    // or
    // PyROOT::(anonymous namespace)::PriorityCmp(PyROOT::PyCallable*, PyROOT::PyCallable*)
    if((ind==0) || sig[ind-1]==':')
      ind=sig.find("(", ind+1);

    while((cidx != string::npos) && (cidx < ind)) {
      cidx = sig.find("::", cstart);
         
      if((cidx != string::npos) && (cidx < ind)){ 
        ++ncolon;
        cstart = cidx+2;
      }
    }
      
    // 15-10-07
    // Be careful with namespaces...
    // things like TBits::TReference::~TReference()
    // will confuse our naive algorithm... instead of just looking
    // for '::', look for the last pair of '::'
    string::size_type icolon = string::npos;
    string::size_type istart = start;
    for (int i=0;i<ncolon;i++){
      if(icolon != string::npos)
        start = istart;
         
      icolon = sig.find("::", istart);
      istart = icolon+2;
    }
      
    if (!isFreeFunc && (icolon != string::npos)) {
      // Dont split it with tokenize because the parameters can have things
      // like std::annoying
      classstr = sig.substr(0, icolon);
      classstr_noname = sig.substr(start, icolon-start);
      protostr = sig.substr(icolon+2, sig.size() - (icolon+2));

      if (gDebug > 0) {
        cerr << "classstr : " << classstr << endl;
        cerr << "protostr : " << protostr << endl;
      }
    }
    else if (!isFreeFunc && (icolon == string::npos)) {
      // We get something without colons when we try to register
      // a class... ignore it
      ++list_iter;
      continue;
    }
    else if( isFreeFunc && (icolon != string::npos)) {
      // What want to register free funcs but this one has ::
      // so we just ignore it
      ++list_iter;
      continue;
    }
    else if( isFreeFunc && (icolon == string::npos)) {
      // This means it's a free function and we have to register ir
      protostr = string(sig);
    }

    if (!isFreeFunc && (classstr.empty() || protostr.empty()) ) {
      if (gDebug > 0)
        cerr << "Couldnt find class or method for symbol : " << sig << endl << endl;
      nerrors++;
      ++list_iter;
      continue;
    }

    // 27-02-08
    // Since there might be collision with the hash function used
    // we can't realy on the hash only... check the name too
    if( !isFreeFunc && (classname != classstr) ){
      ++list_iter;
      continue;
    }

    // 10/05/07
    // this is small hack (yes... again). Let's ignore all the functions
    // that belong to std and start with __ (why? they are weird)
    if ((symbol->fClassHash==hash("std") && classstr=="std") && protostr.at(0)=='_'){
      ++list_iter;
      continue;
    }

    // Here we have to parse it again to detect
    // what we can find between the parenthesis
    string signature = "";
    string::size_type open  = protostr.find('(');
    string::size_type close = protostr.rfind(')');

    if(protostr.find("operator()")!=string::npos)
      open=protostr.find("(", open+1);

    // The name of the method is the proto until the first (
    string methodstr(protostr, 0, open);

    if (methodstr=="operator" && protostr.at(open)=='(' && protostr.at(open+1)==')'){
      if (gDebug > 0)
        cerr << " This thing is tricky (operator). be aware" << endl;
      methodstr += "()";
      open += 2;
    }
    if (gDebug > 0)
      cerr << "methodstr: " << methodstr << endl;

    // Very annoying class... check it later
    if(symbol->IsConst() && methodstr=="TFormulaPrimitive"){
      ++list_iter;
      continue;
    }

    if ( (close - open) > 1)
      signature += protostr.substr(open+1, (close-open)-1 );

    // Here we have a problem....
    // a normal prototype looks like:
    // TH1::Add(TH1 const*, TH1 const*, double, double)
    //
    // but Cint doesnt like this kind of constness
    // the only thing it wants to see is:
    // TH1::Add(const TH1 *, const TH1 *, double, double)
    //
    // Ideally, I should modify Cint to accept both
    // but since I'm only trying to create a proof of concept
    // I will hack my way through for the moment and consider
    // the Cint changes later on.

    // (so... let's change the format for comething more suitable)
    string newsignature = "";
    if (!signature.empty()) {
      string delim(",");
         
      // go to the first pos
      string::size_type lpos = signature.find_first_not_of(delim, 0);
      // first delim
      string::size_type pos  = signature.find_first_of(delim, lpos);

      if(pos == string::npos && lpos != pos)
        pos = signature.size();
      else {
        // 05-11-07
        // We can have something like:
        // Were one of the parameters includes a comma. I would think it's only possible
        // when we have templates and an option is to ignore a comma if it's between "<>"...
        // ROOT::Math::SVector<double, 2u>::operator=(ROOT::Math::SVector<double, 2u> const&)
        string::size_type pos_sm  = signature.find_first_of("<", lpos);
        if(pos_sm != string::npos && pos_sm < pos){
          string::size_type pos_gr  = signature.find_first_of(">", pos_sm);
          if(pos_gr != string::npos && pos < pos_gr && pos > pos_sm)
            pos =  signature.find_first_of(delim, pos_gr); 
        }
            
        if(pos == string::npos && lpos != pos)
          pos = signature.size();
      }

      int k = 0;
      while (pos != string::npos || lpos != string::npos) {
        // new token
        string paramtoken = signature.substr(lpos, pos - lpos);
        string newparam;
		
        // 05-11-07
        // We can get something like:
        // ROOT::Math::SVector<double, 2u>
        // When we have actually declared
        // ROOT::Math::SVector<double,2>
        // in ROOT... so the hashes wont match..
        // try to convert the former in the latter.
            
        string::size_type pos_s  = paramtoken.find_first_of("<", 0);
        string::size_type pos_g  = paramtoken.find_last_of(">", paramtoken.size()-1);

        if( (pos_s != string::npos) &&
            (pos_g != string::npos)){ 

          // ** Pre-process a token
          string::size_type index=0;
          while(index < paramtoken.size()){
            if( index>pos_s && index<pos_g ) {
              if(paramtoken[index]==' ' &&
                 (index>0 && !isalpha(paramtoken[index-1])) && 
                 (index<paramtoken.size()-1 && !isalpha(paramtoken[index+1]))){
                paramtoken.erase(index,1);
                pos_g  = paramtoken.find_last_of(">", paramtoken.size()-1);
              }
              // How is the isinteger(char) when using the stl?
              else if( (paramtoken[index]=='0' || paramtoken[index]=='1' || paramtoken[index]=='2' || paramtoken[index]=='3' || paramtoken[index]=='4' ||
                        paramtoken[index]=='5' || paramtoken[index]=='6' || paramtoken[index]=='7' || paramtoken[index]=='8' || paramtoken[index]=='9')
                       && paramtoken[index+1]=='u'){
                pos_g  = paramtoken.find_last_of(">", paramtoken.size()-1);
                paramtoken.erase(index+1,1);                  
              }
              else
                index++;
            }
            else
              index++;
          }
          // ** Pre-process a token
        }

        if (!paramtoken.empty() && paramtoken.find("const")!=string::npos ) {
          string delim_param(" ");
          string singleparamold;

          // go to the first pos
          string::size_type lpos_param = paramtoken.find_first_not_of(delim_param, 0);
          // first delim
          string::size_type pos_param  = paramtoken.find_first_of(delim_param, lpos_param);
          if(pos_param == string::npos && lpos_param != pos_param)
            pos_param = paramtoken.size();

          // Paramtoken could be something like "TParameter<long long>"
          // in that case the tokenizing by space is wrong!!!
          string::size_type pos_smaller  = paramtoken.find_first_of("<", lpos_param);
              
          // If we find a "<" between the start and the space,
          // then we look for a ">" after the space
          if(pos_smaller != string::npos && pos_smaller < pos_param){
            string::size_type pos_greater  = paramtoken.find_last_of(">", paramtoken.size()-1);
            if(pos_greater != string::npos /*&& pos_greater > pos_param*/)
              pos_param = pos_greater+1;
            else
              cerr << "Error parsing method: " << methodstr << " signature:" << signature << endl;
          }

          int l = 0;
          while (pos_param != string::npos || lpos_param != string::npos) {
            // new token
            string singleparam = paramtoken.substr(lpos_param, pos_param - lpos_param);
            if (l > 0 && singleparam == "const*") {
              newparam += "const ";
              newparam += singleparamold;
              newparam += " *";
            }
            else if (l > 0 && singleparam == "const&") {
              newparam += "const ";
              newparam += singleparamold;
              newparam += " &";
            }
            else if (l > 0 && singleparam == "const*&") {
              newparam += "const ";
              newparam += singleparamold;
              newparam += " *&";
            }
            else if (l > 0 && singleparam == "const**") {
              newparam += "const ";
              newparam += singleparamold;
              newparam += " **";
            }
            // skip delim
            lpos_param = paramtoken.find_first_not_of(delim_param, pos_param);
            // find next token
            pos_param  = paramtoken.find_first_of(delim_param, lpos_param);
            if(pos_param == string::npos && lpos_param != pos_param)
              pos_param = paramtoken.size();
 
            singleparamold = singleparam;
            ++l;
          }
        }
        else if(!paramtoken.empty() && paramtoken.find("const")==string::npos)
          newparam = paramtoken;

        if(k > 0)
          newsignature += ",";

        if(newparam.empty())
          newsignature += paramtoken;
        else
          newsignature += newparam;
           
        // skip delim
        lpos = signature.find_first_not_of(delim, pos);
        // find next token
        pos  = signature.find_first_of(delim, lpos);
         
        if(pos == string::npos && lpos != pos)
          pos = signature.size();
        ++k;
      }
    }
    if (gDebug > 0) {
      cerr << "--- OLD Signature: " << signature << endl;
      cerr << "--- NEW Signature: " << newsignature << endl;
    }

    // 21/05/7
    // We have another problem, certain symbols have names like: 
    // shared_ptr<Base>* const shared_ptr<Base>::n3<Base>(shared_ptr<Base> const&)
    // which obviously doesnt give us a good name for a class. And as far as I can tell,
    // only represents the return type... so lets strip it out to obtain only
    // the name of the class
    string::size_type ispace  = classstr.rfind(" ", 0);

    string finalclass;
    if(ispace != string::npos){
      // Let's just assume this is the last entry (can a classname contain spaces? )
      finalclass = signature.substr(ispace, classstr.size());
    }
    else
      finalclass = classstr;


    // 12-10-07
    // CInt doesn't believe that a constructor can be different from the name of
    // the class. So when we have things like:
    //
    // TParameter<double>::TParameter() 
    //
    // CInt will just think it's 
    // 
    // TParameter<double>::TParameter<double>()
    // 
    // Changint this in CInt would probably requiere more changes than changing the
    // real name to what Cint expects.
      
    // 11-10-07
    // Get rid of the "<>" part in something like TParameter<float>::TParameter()
    if(symbol->IsConst()){
      string::size_type itri = finalclass.find("<");
      
      if(itri != string::npos){
        methodstr = finalclass;
      }
    }
    else if(symbol->IsDest()){
      string::size_type itri = finalclass.find("<");
      
      if(itri != string::npos) {
        methodstr = "~" + finalclass;
      }
    }

    // 31-07-07
    // I forgot something else....
    // the stupid constness (rem two functions can vary only by their constness)
    // so lets check it here now
    int isconst=0;

    string::size_type lpar = sig.find_last_of(")");
    string::size_type pos_par  = sig.find_first_of("const", lpar);
    if(pos_par != string::npos)
      isconst=1;

    // 18-10-07
    // How can we deal with things like:
    // 
    // TClass::GetClass(char const*, bool)::full_string_name
    // TClass::GetClass(char const*, bool)::__PRETTY_FUNCTION__
    if(sig.find_first_of("::", lpar)!=string::npos){
      ++list_iter;
      continue;
    }

    // 16-15-07
    // Another ugly hack
    // deal with annoying things that don't even look like a function:
    // (This has to be solved soon)
    /*
      void (*std::for_each<__gnu_cxx::__normal_iterator<PyROOT::(anonymous namespace)::PyError_t*, std::vector<PyROOT::(anonymous namespace)::PyError_t, std::allocator<PyROOT::(anonymous namespace)::PyError_t> > >, void (*)(PyROOT::(anonymous namespace)::PyError_t&)>(__gnu_cxx::__normal_iterator<PyROOT::(anonymous namespace)::PyError_t*, std::vector<PyROOT::(anonymous namespace)::PyError_t, std::allocator<PyROOT::(anonymous namespace)::PyError_t> > >, __gnu_cxx::__normal_iterator<PyROOT::(anonymous namespace)::PyError_t*, std::vector<PyROOT::(anonymous namespace)::PyError_t, std::allocator<PyROOT::(anonymous namespace)::PyError_t> > >, void (*)(PyROOT::(anonymous namespace)::PyError_t&)))(PyROOT::(anonymous namespace)::PyError_t&)
    */
    if(sig.find("PyError_t")!=string::npos) {
      ++list_iter;
      continue;
    }
      
    char *tmpstr = new char[classname.size()+56]; // it could be changed by G__register_pointer and the size can be bigger
    strcpy(tmpstr, classname.c_str());
    if( G__register_pointer(tmpstr, methodstr.c_str(), newsignature.c_str(), symbol->fMangled,isconst) == -1) {
      // yahoo.... we can finally call our register method
      if (gDebug > 0) {
        cerr << "xxx Couldnt register the method: " << methodstr << endl;
        cerr << "xxx from the class    : " << finalclass << endl;
        cerr << "xxx classname    : " << classname << endl;
        cerr << "xxx with the signature: " << newsignature << endl;
      }
      nerrors++;
    }
    else {
      if (gDebug > 0) {
        cerr << " *** Method registered  : " << methodstr << endl;
        cerr << " *** from the class     : " << finalclass << endl;
        cerr << " *** with the signature : " << newsignature << endl;
      }
      nreg++;
    }
    delete [] tmpstr;

    ++list_iter;
    demangled->remove(symbol);
    delete symbol;
  }
  if (gDebug > 0)
    cerr << "****************************************" << endl << endl;

  if (gDebug > 0) {
    cerr << "*************************************" << endl;
    cerr << " Number of symbols registered : " << nreg << endl;
    cerr << " Number of errors             : " << nerrors << endl;
    cerr << "*************************************" << endl << endl;
  }
}
