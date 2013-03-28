/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file bc_parse.cxx
 ************************************************************************
 * Description:
 *  block scope parser and compiler
 ************************************************************************
 * Copyright(c) 2004~2005  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "bc_parse.h"
#include "bc_inst.h"
#include "bc_reader.h"

#include <deque>

/***********************************************************************
 * static object
 ***********************************************************************/
G__blockscope *G__currentscope;

/***********************************************************************
 * G__breaktable
 ***********************************************************************/
void G__breaktable::resolve(G__bc_inst& inst,int destination) {
  for(vector<int>::iterator i=m_breaktable.begin();i!=m_breaktable.end();++i) {
    inst.Assign(*i,destination);  
  }
}

/***********************************************************************
 * G__gototable
 ***********************************************************************/
void G__gototable::resolve(G__bc_inst& inst) {
  map<string,int>::iterator i;
  int destination;
  int origin;
  for(i=m_gototable.begin();i!=m_gototable.end();++i){
    origin = (*i).second;
    destination = m_labeltable[(*i).first];
    if(!destination) {
      //error?;
      G__fprinterr(G__serr,"Error: label '%s' not found",(*i).first.c_str());
      G__genericerror((char*)NULL);
    }
    inst.Assign(origin,destination);
  }
}


/***********************************************************************
 * G__blockscope
 ***********************************************************************/

////////////////////////////////////////////////////////////////////////////
// ctor/dtor
////////////////////////////////////////////////////////////////////////////
/***********************************************************************
 * G__blockscope::G__blockscope()
 ***********************************************************************/
G__blockscope::G__blockscope():
   m_ifunc(0),
   m_iexist(-1),
   m_var(0),
   store_p_local(0),
   m_preader(0),
   isvirtual(0),
   isstatic(0),
   m_pcasetable(0),
   m_pbreaktable(0),
   m_pcontinuetable(0),
   m_pgototable(0)
{
  // do nothing, should be initialized later by Init()
}

/***********************************************************************
 * G__blockscope::G__blockscope()
 ***********************************************************************/
G__blockscope::G__blockscope(G__blockscope* enclosing):
   m_ifunc(0),
   m_iexist(-1),
   m_var(0),
   store_p_local(0),
   m_preader(0),
   isvirtual(0),
   isstatic(0),
   m_pcasetable(0),
   m_pbreaktable(0),
   m_pcontinuetable(0),
   m_pgototable(0)
{
  Init(enclosing);
}

/***********************************************************************
 * G__blockscope::~G__blockscope()
 ***********************************************************************/
G__blockscope::~G__blockscope() {
  G__p_local = store_p_local;
  // free((void*)m_var); // should not free m_var, it is stored in bytecode
}

/***********************************************************************
 * G__blockscope::G__blockscope()
 ***********************************************************************/
void G__blockscope::Init(G__blockscope* enclosing) {
  // reset jump tables

  // set enclosing/enclosed scope info to m_var
  m_var = (struct G__var_array*)malloc(sizeof(struct G__var_array));
  memset(m_var,0,sizeof(struct G__var_array));
  m_var->tagnum = -1;
  store_p_local = G__p_local;
  G__p_local = m_var;

  if(enclosing) {
    m_pcasetable = enclosing->m_pcasetable;
    m_pbreaktable = enclosing->m_pbreaktable;
    m_pcontinuetable = enclosing->m_pcontinuetable;
    m_pgototable = enclosing->m_pgototable;
    m_preader = enclosing->m_preader;
    m_ifunc = enclosing->m_ifunc;
    m_iexist = enclosing->m_iexist;
    m_bc_inst = enclosing->m_bc_inst;
    m_var->enclosing_scope = enclosing->m_var;
    m_var->tagnum = enclosing->m_var->tagnum;
    int i=0;
    struct G__var_array *var = enclosing->m_var;;
    if(var->inner_scope) {
      while(var->inner_scope[i]) ++i;
      var->inner_scope = 
       (struct G__var_array**)realloc((void*)var->inner_scope,sizeof(void*)*(i+2));
    }
    else {
      i=0;
      var->inner_scope = (struct G__var_array**)malloc(sizeof(void*)*(i+2));
    }
    var->inner_scope[i] = m_var;
    var->inner_scope[i+1] = 0;
  }
  else {
    m_pcasetable = 0;
    m_pbreaktable = 0;
    m_pcontinuetable = 0;
    m_pgototable = 0;
  }

  // in turn, this sets G__p_ifunc->ifunc and ifn
  // but this implementation is questionable.
  m_var->ifunc = m_ifunc;
  m_var->ifn=m_iexist;
}

////////////////////////////////////////////////////////////////////////////
// 1st level
////////////////////////////////////////////////////////////////////////////

/***********************************************************************
 * G__blockscope::compile()
 ***********************************************************************/
int G__blockscope::compile(int openBrace) {
  m_bc_inst.ENTERSCOPE(); 
  int c = compile_core(openBrace);
  m_bc_inst.EXITSCOPE(); 
  return(c);
}

/***********************************************************************
 * G__blockscope::compile_core()
 ***********************************************************************/
int G__blockscope::compile_core(int openBrace) {
  // openBrace==1 {  statements;     }
  // openBrace==0   {  statements;   }
  // openBrace==0   statement        ;  
  //               ^ ---------------> ^
  int c = 0;
  string token;

  for(;;) {

    if(c==0xff) c = m_preader->fgetc_gettoken();
    else if(c==0) {
      c = m_preader->fappendtoken(token,c);
      m_bc_inst.CL();
    }
    else        c = m_preader->fappendtoken(token,c);

    switch( c ) {
    case ' ' : /* space */
    case '\t' : /* tab */
    case '\n': /* end of line */
    case '\r': /* end of line */
    case '\f': /* end of line */
      c = compile_space(token,c); // always finish
      break;

    case '+': 
    case '-': 
    case '%': 
    //case '>': 
    case '!': 
    case '=': 
    case '?': 
    case '.': 
    case '/': // <<< new, comment is handled in fgetc_gettoken()
    case '^': 
      c=compile_operator(token,c); // always finish
      break;

    case '&':
    case '*': 
      c=compile_operator_AND_ASTR(token,c); // always finish
      break;

    case '<': 
      c=compile_operator_LESS(token,c); // finish or continue
      break;

#ifdef G__NEVER
    case '/':  
      // this case should be merged to compile_operator, since comment is
      // handled in G__reader::fgetc_gettoken()
      c=compile_operator_DIV(token,c); // always finish
#endif

    case '(': 
      c=compile_parenthesis(token,c); // always finish
      break;

    case '[': 
      c=compile_bracket(token,c); // always finish
      break;

    case '{' :
      if(!openBrace && token=="") {
        // openBrace==0   {  statements;   }
        //                 ^
        openBrace=1;
        c=0;
      }
      else {
        // openBrace==1     { {   } }
        // openBrace==X   do  {   }
        // openBrace==X   try {   }
        //                     ^
        c = compile_brace(token,c); // always finish
      }
      break;

    case '}' :
      // end of block
      // openBrace==1 {  statements;     }
      // openBrace==0   {  statements;   }
      // openBrace==0   statement        ;  
      //               ^ ---------------> ^
      return(c);

    case ':': 
      c = compile_column(token,c); // finish or continue
      break;

    case ';': 
      c = compile_semicolumn(token,c); // always finish
      break;

    case ',': 
      // expr , expr
      //       ^
      compile_expression(token);
      break;

#if G__NEVER
    case '#' : // Preprocessor symbol is handled in G__reader class
      c = compile_preprocessor(token,c); // always finish
      break;
#endif

    case '"': 
    case '\'': 
      // this should not happen, however a leagal C++ expression
      // string contant can never be lvalue of expression.
      break;

    case EOF:
    case ']': 
    case ')': 
    default:
      // error;
      G__fprinterr(G__serr,"Error: Syntax error '%s %c'",token.c_str(),c);
      G__genericerror((char*)NULL);
      break;
    }

    if(c==';'|| c=='}') {
      if(!openBrace) break;
      else c=0;
    }
  }
  return(c);
}

////////////////////////////////////////////////////////////////////////////
// expression, 2nd level
////////////////////////////////////////////////////////////////////////////
/***********************************************************************
 * G__blockscope::compile_expression()
 *	func(args);  -> expr
 *	type(args);  -> expr
 *      (expr);      -> expr
 *      (cast)expr;  -> expr
 *      expr;        -> expr
 *      expr/expr;   -> expr
 *      expr<expr;   -> expr
 *      expr,expr;   -> expr
 *      expr&expr;   -> expr
 *      expr*expr;   -> expr
 ***********************************************************************/
////////////////////////////////////////////////////////////////////////////
G__value G__blockscope::compile_expression(string& expr) {
  size_t len = expr.size()+1;
  char *buf = new char[len];
  strncpy(buf,expr.c_str(), len);
  if(expr.size()>G__LONGLINE) {
    G__fprinterr(G__serr,"Limitation: Expression is too long %d>%d %s "
	         ,len,G__LONGLINE,buf);
    G__genericerror((char*)NULL);
  }
  G__blockscope *store_scope = G__currentscope;
  int store_var_type = G__var_type;
  G__var_type = 'p';
  G__currentscope = this;
  G__value x = G__getexpr(buf); // legacy
  G__currentscope = store_scope;
  G__var_type = store_var_type;
  stdclear(expr);
  delete [] buf;
  return(x);
}

////////////////////////////////////////////////////////////////////////////
G__value G__blockscope::compile_arglist(string& args,G__param* libp) {
  // args = "expr,expr,expr"

  // todo, This implementation of replacing reader object is not so clean.
  //       need to review this.
  G__srcreader<G__sstream> stringreader;
  stringreader.Init(args.c_str());
  
  int c=0;
  string expr;
  libp->paran = 0;

  do {
    c = stringreader.fgetstream(expr,",");
    if(expr.size()) {
      libp->para[libp->paran++] = compile_expression(expr);
    }
  } while(c==',');

  libp->para[libp->paran] = G__null;

  return(G__null);
}

////////////////////////////////////////////////////////////////////////////
int G__blockscope::getstaticvalue(string& expr) {
  int store_asm_noverflow = G__asm_noverflow;
  int store_no_exec_compile = G__no_exec_compile;
  size_t len = expr.size()+1;
  char *buf = new char[len];
  strncpy(buf,expr.c_str(), len);
  if(expr.size()>G__LONGLINE) {
    G__fprinterr(G__serr,"Limitation: Expression is too long %d>%d %s "
	         ,expr.size(),G__LONGLINE,buf);
    G__genericerror((char*)NULL);
  }
  G__asm_noverflow = 0;
  G__no_exec_compile = 0;
  int result = G__int(G__getexpr(buf)); // legacy
  delete [] buf;
  G__no_exec_compile = store_no_exec_compile;
  G__asm_noverflow = store_asm_noverflow;
  return(result);
}

////////////////////////////////////////////////////////////////////////////
// space, 2nd level
////////////////////////////////////////////////////////////////////////////

/***********************************************************************
 * G__blockscope::compile_space()
 * static type x;
 * const type x;
 * mutable type x;
 * register type x;
 * volatile type x;
 * storage type x;
 * class a x;
 * struct a x;
 * enum a x;
 * union a x;
 * type x;
 * type f();
 * virtual type f(); //never happen in function
 * goto label;
 * return val;
 * case x:
 * throw x;
 * delete x;
 ***********************************************************************/
int G__blockscope::compile_space(string& token,int c) {
  if(token=="case")        c = compile_case(token,c);     // c==0
  else if(token=="new")    c = compile_new(token,c);    // c==';'
  else if(token=="delete") c = compile_delete(token,c,0); // c==';'
  else if(token=="throw")  c = compile_throw(token,c);    // c==';'
  else if(token=="goto") {
    stdclear(token);
    c = m_preader->fgetstream(token,";" /* ,0 */ );  // c==';'
    //m_pgototable->addgoto(m_bc_inst.GetPC(),token);
    m_pgototable->addgoto(m_bc_inst.JMP(),token);
    stdclear(token);
  }
  else if(token=="return") {
    stdclear(token);
    c = compile_return(token,c);
  }
  else {
    // type x;
    // storagekeyword typedecorator type x;
    G__TypeReader type;
    while(type.append(token,c)) { 
      c = m_preader->fgettoken(token);
    }
    if(!type.Type()) {
      // error
      G__fprinterr(G__serr,"Error: type '%s' undefined",token.c_str());
      G__genericerror((char*)NULL);
    }
    c = compile_declaration(type,token,c); // c==';'
  }
  return(c);
}

////////////////////////////////////////////////////////////////////////////
int G__blockscope::compile_case(string& token,int c) {
  // case expression :
  //      ^ ------->  ^
  c = m_preader->fgetstream(token,":" /* ,0 */ );
  int val = getstaticvalue(token);
  m_pcasetable->addcase(val,m_bc_inst.GetPC());
  stdclear(token);
  c=0;
  return(c); // c==0
}

////////////////////////////////////////////////////////////////////////////
int G__blockscope::compile_default(string& token,int c) {
  m_pcasetable->adddefault(m_bc_inst.GetPC());
  stdclear(token);
  c=0;
  return(c); // c==0
}

////////////////////////////////////////////////////////////////////////////
int G__blockscope::compile_new(string& token,int c) {
  return(compile_operator(token,c));
}

////////////////////////////////////////////////////////////////////////////
int G__blockscope::compile_delete(string& token,int c,int /* isarray */) {
  // delete    x;
  // delete [] x;
  //          ^
  string expr;
  c = m_preader->fgetstream(expr,";");
  if(token=="delete") {
    stdclear(token);
    compile_deleteopr(expr,0);
  }
  else if(token=="delete[]") {
    stdclear(token);
    compile_deleteopr(expr,1);
  }
  else {
    // error
    G__fprinterr(G__serr,"Error: Syntax error '%s'",token.c_str());
    G__genericerror((char*)NULL);
  }

  return(c);
}

////////////////////////////////////////////////////////////////////////////
// operator, 2nd level
////////////////////////////////////////////////////////////////////////////
/***********************************************************************
 * G__blockscope::compile_operator()
 ***********************************************************************/
int G__blockscope::compile_operator(string& token,int c) {
  string buf;
  if(c) token.append((string::size_type)1,(char)c);
  c = m_preader->fgetstream(buf,";" , (c=='('?1:0)  ); // c==';'
  token.append(buf);
  compile_expression(token);
  return(c); // c==';'
}

////////////////////////////////////////////////////////////////////////////
// operator(), 2nd level
// func    ();     -> expr
// macro   ()      -> 
//         (expr);      -> expr
//         (cast)expr;  -> expr
//          ^
////////////////////////////////////////////////////////////////////////////
/***********************************************************************
 * G__blockscope::compile_operator_PARENTHESIS()
 ***********************************************************************/
int G__blockscope::compile_operator_PARENTHESIS(string& token,int c) {
  string buf;
  if(c) token.append((string::size_type)1,(char)c);
  c = m_preader->fgetstream_(buf,";" , 0 ); // c==')'
  token.append(buf);
  if(c) token.append((string::size_type)1,(char)c);
  c = m_preader->fgetspace();
  if(c==';') {
    compile_expression(token);
    stdclear(token);
    return(c); // c==';'
  }
  else if(c==',' ) {
    do {
      compile_expression(token);
      c = m_preader->fgetstream(buf,",;" , 0 ); // c==')'
      token = buf;
    } while(c==',') ; // c==',' || ';'
    compile_expression(token);
    stdclear(token);
    return(c); // c==';'
  }
  else if(G__isoperator(c) || c=='.' || c=='[') {
    token.append((string::size_type)1,(char)c);
    c = m_preader->fgetstream(buf,";" , 0 ); // c==';'
    token.append(buf);
    compile_expression(token);
    stdclear(token);
    return(c); // c==';'
  }
  else {
    int iout=0;
    size_t len = token.size()+10;
    char *str = (char*)malloc(len);
    strncpy(str,token.c_str(), len);
    m_preader->putback();
    G__execfuncmacro(str,&iout); // legacy
    free((void*)str);
    stdclear(token);
    c=';';
    return(c);
  }
}

////////////////////////////////////////////////////////////////////////////
// operator * &, 2nd level
//  type*
//  type&
//  *expr
//  &expr
//  expr*
//  expr*=
//  expr&
////////////////////////////////////////////////////////////////////////////
/***********************************************************************
 * G__blockscope::compile_operator_AND_ASTR()
 ***********************************************************************/
int G__blockscope::compile_operator_AND_ASTR(string& token,int c) {
  if(token=="return"){
    stdclear(token);
    if(c) token.append((string::size_type)1,(char)c);
    c = compile_return(token,c);
  }
  else if(token=="throw"){
    stdclear(token);
    if(c) token.append((string::size_type)1,(char)c);
    c = compile_throw(token,c);
  }
  else if(token=="delete"){
    stdclear(token);
    if(c) token.append((string::size_type)1,(char)c);
    c = compile_delete(token,c,0);
  }
  else if(Istypename(token)) {
    // type*& var;
    // type* const * & var;
    G__TypeReader type;
    while(type.append(token,c)) {
      c = m_preader->fgettoken(token);
    }
    c = compile_declaration(type,token,c); // c==';'
  }
  else {
    c = compile_operator(token,c); // c==';'
  }
  return(c); // c==';'
}

////////////////////////////////////////////////////////////////////////////
// operator < , 2nd level
//  template<class ..>
//  tmplt   <tmparg> var;
//  tmplt   <tmplt<tmparg> > var;             -> declaration
//  tmplt   <tmparg>::enclosedclass::member;  -> expr
//  tmplt   <tmparg>(arg);
//  expr    <expr
//           ^
////////////////////////////////////////////////////////////////////////////
/***********************************************************************
 * G__blockscope::compile_operator_LESS()
 ***********************************************************************/
int G__blockscope::compile_operator_LESS(string& token,int c) {
  if(token=="template") {
    // template<class ..>
    G__declare_template(); // legacy 
    c = ';'; // ??
  }
  else if(G__defined_templateclass((char*)token.c_str())) { // legacy
    // tmplt<tmparg> var                     -> decl
    // tmplt<tmparg>*& var                   -> decl
    // tmplt<tmparg>(arg);                   -> expr
    // tmplt<tmparg>::type var;              -> decl
    // tmplt<tmparg>::type(arg);             -> expr
    // tmplt<tmparg>::staticmember;          -> expr
    // tmplt<tmparg>::staticmemfunc(arg);    -> expr
    //       ^ ---> ^
    token.append((string::size_type)1,(char)c);
    string buf;
    c= m_preader->fgetstream_template(buf,">" /* ,1 */ );
    token.append(buf);
    token.append((string::size_type)1,(char)c);

    c = 0xff; // special flag to continue

  }
  else {
    // expr<expr;
    c = compile_operator(token,c); // c==';'
  }
  return(c); // c==';' or c==0
}

////////////////////////////////////////////////////////////////////////////
// operator / , 2nd level
//	expr/expr
//	expr/=expr
//	C++ style comment
//	C style comment
////////////////////////////////////////////////////////////////////////////
/***********************************************************************
 * G__blockscope::compile_operator_DIV()
 ***********************************************************************/
// this function shouldn't be needed 
// since comment is handled in G__reader::fgetc_gettoken()
int G__blockscope::compile_operator_DIV(string& token,int c) {
  c = m_preader->fgetc();
  switch(c) {
  case '/':  // C++ style comment
    m_preader->fignoreline(); 
    c = 0;
    break;
  case '*':  // C style comment
    m_preader->skipCcomment();
    c = 0;
    break;
  default:
    m_preader->putback();
    c = '/';
    c = compile_operator(token,c); // c==';'
    break;
  }
  return(c);
}

////////////////////////////////////////////////////////////////////////////
// operator [] , 2nd level
//  var[i] = x; expr
//  delete[] x;
////////////////////////////////////////////////////////////////////////////
/***********************************************************************
 * G__blockscope::compile_bracket()
 ***********************************************************************/
int G__blockscope::compile_bracket(string& token,int c) {
  if(token=="delete") {
    // delete [] x;
    //         ^
    c = m_preader->fappendtoken(token,c);
    if(c) token.append((string::size_type)1,(char)c);
    c=0;
    if(token!="delete[]") {
      G__fprinterr(G__serr,"Error: Syntax error '%s'",token.c_str());
      G__genericerror((char*)NULL);
    }
    c = compile_delete(token,c,1);     // c==';'
  }
  else {
    // a[x];
    c = compile_operator(token,c);  // c==';'
  }
  return(c);
}

////////////////////////////////////////////////////////////////////////////
// operator :
//      default:
//      public:    // never
//      private:    // never
//      protected:    // never
//      scope1::member; -> expr
//      scope1::type x; -> declaration
//      label:
////////////////////////////////////////////////////////////////////////////
/***********************************************************************
 * G__blockscope::compile_column()
 ***********************************************************************/
int G__blockscope::compile_column(string& token,int c) {
  if(token=="default")        c = compile_default(token,c);
  else if(token=="public")    stdclear(token);
  else if(token=="protected") stdclear(token);
  else if(token=="private")   stdclear(token);
  else {
    //
    c = m_preader->fgetc();
    if(c==':') {
      // scope1::type var;             -> decl
      // scope1::type*& var;           -> decl
      // scope1::type(arg);            -> expr
      // scope1::staticmember;         -> expr
      // scope1::staticmemfunc(arg);   -> expr
      //         ^
      token.append("::");
      c=0;
      // return to compile() and continue
    }
    else {
       // label:
       m_preader->putback();
       m_pgototable->addlabel(token,m_bc_inst.GetPC());
       stdclear(token);
       c=0;
    }
  }
  return(c);
}

////////////////////////////////////////////////////////////////////////////
// ';'
//      return;
//      continue;
//      break;
//      expr;
////////////////////////////////////////////////////////////////////////////
/***********************************************************************
 * G__blockscope::compile_semicolumn()
 ***********************************************************************/
// break;
// continue;
// return;
// expr;
int G__blockscope::compile_semicolumn(string& token,int c) {
  if(token=="break") {
    m_pbreaktable->add(m_bc_inst.JMP());
    stdclear(token);
  }
  else if(token=="continue") {
    m_pcontinuetable->add(m_bc_inst.JMP());
    stdclear(token);
  }
  else if(token=="return") {
    m_bc_inst.RTN_FUNC(0);
    stdclear(token);
  }
  else if(strncmp(token.c_str(),"return\"",7)==0 ||
	  strncmp(token.c_str(),"return'",7)==0) {
    string val = token.substr(6);
    compile_expression(val);
    m_bc_inst.RTN_FUNC(1);
    stdclear(token);
  }
  else if(token=="throw")    compile_throw(token,c);  // c==';'
  else                       compile_expression(token);
  return(c);
}


////////////////////////////////////////////////////////////////////////////
// parenthesis, 2nd level
////////////////////////////////////////////////////////////////////////////

/***********************************************************************
 * G__blockscope::compile_parenthesis()
 *	if(expr)
 *	while(expr)
 *	for(expr;expr;expr)
 *      switch(expr)
 *	return(expr)
 *      throw(expr); 
 *      catch(expr); 
 *	func(args);  -> expr
 *	type(args);  -> expr
 *      (expr);      -> expr
 *      (cast)expr;  -> expr
 ***********************************************************************/
int G__blockscope::compile_parenthesis(string& token,int c) {
  if(token=="if") {
    c = compile_if(token,c);  // c==';' or '}'
  }
  else if(token=="for") {
    G__blockscope forscope(this);
    c = forscope.compile_for(token,c);  // c==';' or '}'
  }
  else if(token=="while") {
    c = compile_while(token,c);  // c==';' or '}'
  }
  else if(token=="switch") {
    c = compile_switch(token,c);  // c=='}'
  }
  else if(token=="return") {
    c = compile_return(token,c);  // c==';'
  }
  else if(token=="throw") {
    c = compile_throw(token,c);  // c==';'
  }
  else if(token=="catch") {
    // error, this is not allowed
    G__fprinterr(G__serr,"Error: 'catch' appears without 'try'");
    G__genericerror((char*)NULL);
    c = compile_catch(token,c);  // c=='}'
  }
  else if(token=="operator") {
    // operator()();   -> expr
    //          ^
    c = compile_operator(token,c);  // c==';'
  }
  else {
    // macro   ()           -> macro
    // func    ();          -> expr
    //         (expr);      -> expr
    //         (cast)expr;  -> expr
    //          ^
    c = compile_operator_PARENTHESIS(token,c);  // c==';'
  }
  return(c);
}

////////////////////////////////////////////////////////////////////////////
// parenthesis, 3rd level
////////////////////////////////////////////////////////////////////////////

/***********************************************************************
 * G__blockscope::compile_if()
 *  if(expr) { } else { }
 *     ^
 ***********************************************************************/
int G__blockscope::compile_if(string& token,int c) {

  // Pseudo code

  G__bc_pointer_addr pc_else;
  G__bc_pointer_addr pc_end;

  // if(expr) 
  //    ^--> ^
  stdclear(token);
  c = m_preader->fgetstream(token,")"  /* ,1 */ );
  compile_expression(token);
  pc_else = m_bc_inst.CNDJMP(0);

  //  { true clause } 
  G__blockscope trueblock(this);
  c = trueblock.compile();

  // if there is else ?
  m_preader->storepos(c);
  string buf;
  c = m_preader->fgettoken(buf);
  //c = m_preader->fgetstream(buf,"{(" /* ,0 */ ); // ??? fgettoken ???


  if(buf=="else") {
    pc_end = m_bc_inst.JMP(0);
    m_bc_inst.Assign(pc_else,m_bc_inst.GetPC());
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,">> %3x: JMP %x\n",pc_else-1,m_bc_inst.GetPC());
#endif
    G__blockscope falseblock(this);
    c = falseblock.compile((c=='{')?1:0);
    m_bc_inst.Assign(pc_end,m_bc_inst.GetPC());
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,">> %3x: JMP %x\n",pc_end-1,m_bc_inst.GetPC());
#endif
  }
  else {
    m_bc_inst.Assign(pc_else,m_bc_inst.GetPC());
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,">> %3x: JMP %x\n",pc_else-1,m_bc_inst.GetPC());
#endif
    c = m_preader->rewindpos();
  }

  return(c);
}

/***********************************************************************
 * G__blockscope::compile_switch()
 *  switch(expr) { case a:  break; default: break; }
 *         ^
 ***********************************************************************/
int G__blockscope::compile_switch(string& token,int c) {
  G__breaktable breaktable;
  G__casetable *pcasetable = new G__casetable;

  G__blockscope block(this);

  block.setbreaktable(&breaktable);
  block.setcasetable(pcasetable);

  // read expression
  stdclear(token);
  // switch(expr) { case a:  break; default: break; }
  //        ^--> ^
  c = m_preader->fgetstream(token,")" /* ,1 */ );
  compile_expression(token);

  // 
  m_bc_inst.CASE(pcasetable);

  // compile switch block
  c = block.compile();

  breaktable.resolve(m_bc_inst,m_bc_inst.GetPC());

  // delete pcasttable; // pcasttable is part of bytecode, 
                        // it is deleted together with bytecode.
  return(c);
}

/***********************************************************************
 * G__blockscope::compile_for()
 *  for(expr;expr;expr) {    break;  }
 *      ^
 ***********************************************************************/
int G__blockscope::compile_for(string& token,int c) {
  G__breaktable breaktable;
  G__breaktable continuetable;
  int addr_iter;
  int addr_next;
  int addr_end;

  G__blockscope block(this);
  block.setbreaktable(&breaktable);
  block.setcontinuetable(&continuetable);

  // read initialization,  for(expr;expr;expr) {    break;  }
  //                           ^^^^^
  c = compile_core(); // c==';'
  //stdclear(token);
  //c = m_preader->fgetstream(token,";" /* ,0 */ );
  //compile_expression(token); // this should be in forblock and has declaration

  // loop condition,  for(expr;expr;expr) {    break;  }
  //                           ^^^^^
  addr_iter = m_bc_inst.GetPC();
  stdclear(token);
  c = m_preader->fgetstream(token,";" /* ,0 */ );
  if(token!="") {
    compile_expression(token);
    breaktable.add(m_bc_inst.CNDJMP());
  }

  // read increment instruction,  for(expr;expr;expr) {    break;  }
  //                                            ^^^^^
  c = m_preader->fgetstream(token,")" /* ,1 */);

  // compile for block,  for(expr;expr;expr) {    break;  }
  //                                            ^^^^^^^^^^^^^^
  c = block.compile();

  addr_next = m_bc_inst.GetPC();

  // compile increment instruction,  for(expr;expr;expr) {    break;  }
  //                                               ^^^^^
  compile_expression(token);
  m_bc_inst.JMP(addr_iter);
  addr_end = m_bc_inst.GetPC();

  // resolve jump tables
  continuetable.resolve(m_bc_inst,addr_next);
  breaktable.resolve(m_bc_inst,addr_end);

  // loop optimization
  m_bc_inst.optimizeloop(addr_iter,addr_end);

  return(c);
}

/***********************************************************************
 * G__blockscope::compile_while()
 * while(expr) {    break;  }
 *       ^
 ***********************************************************************/
int G__blockscope::compile_while(string& token,int c) {
  G__breaktable breaktable;
  G__breaktable continuetable;
  int addr_iter;
  int addr_next;
  int addr_end;

  G__blockscope block(this);
  block.setbreaktable(&breaktable);
  block.setcontinuetable(&continuetable);

  // loop condition,  while(expr) {    break;  }
  //                        ^ -> ^
  addr_iter = m_bc_inst.GetPC();
  addr_next = addr_iter;
  stdclear(token);
  c = m_preader->fgetstream(token,")" /* ,1 */ );
  compile_expression(token);
  breaktable.add(m_bc_inst.CNDJMP());

  // compile switch block,  while(expr) {    break;  }
  //                                    ^^^^^^^^^^^^^^
  c = block.compile();

  // compile increment instruction, while(expr)   {    break;  }
  //                                                          ^^^
  m_bc_inst.JMP(addr_iter);
  addr_end = m_bc_inst.GetPC();

  // resolve jump tables
  continuetable.resolve(m_bc_inst,addr_next);
  breaktable.resolve(m_bc_inst,addr_end);

  // loop optimization
  m_bc_inst.optimizeloop(addr_iter,addr_end);

  return(c);
}

/***********************************************************************
 * G__blockscope::compile_do()
 *  do {    break;  } while(expr);
 *      ^
 ***********************************************************************/
int G__blockscope::compile_do(string& token,int c) {
  G__breaktable breaktable;
  G__breaktable continuetable;
  int addr_iter;
  int addr_next;
  int addr_end;

  G__blockscope block(this);
  block.setbreaktable(&breaktable);
  block.setcontinuetable(&continuetable);

  addr_iter = m_bc_inst.GetPC();

  // compile block, do {    break; } while(expr);
  //                    ^^^^^^^^^^^^
  c = block.compile(1);
  addr_next = m_bc_inst.GetPC();

  // do {    break; } while(expr);
  //                 ^ ---> ^
  stdclear(token);
  c = m_preader->fgetstream(token,"(" /* ,0 */ );

  // do {    break; } while(expr);
  //                        ^ -> ^
  stdclear(token);
  c = m_preader->fgetstream(token,")" /* ,1 */ );
  compile_expression(token);
  m_bc_inst.CND1JMP(addr_iter);
  addr_end = m_bc_inst.GetPC();

  // do {    break; } while(expr)  ;
  //                             ^->^
  c = m_preader->fignorestream(";");

  // resolve jump tables
  continuetable.resolve(m_bc_inst,addr_next);
  breaktable.resolve(m_bc_inst,addr_end);

  // loop optimization
  m_bc_inst.optimizeloop(addr_iter,addr_end);

  return(c);
}

/***********************************************************************
 * G__blockscope::compile_return()
 *  return(val);
 *  return val ;
 *         ^ -> ^
 *  return(*this)[2];
 *         ^ ------> ^
 ***********************************************************************/
int G__blockscope::compile_return(string& token,int c) {
  stdclear(token);
  int c2;
  c2 = m_preader->fgetstream(token,";" /* ,(c=='(')?1:0 */ );
  string expr;
  if(c=='(') expr = string("(") + token;
  else if(c=='"') expr = string("\"") + token;
  else if(c=='\'') expr = string("'") + token;
  else       expr = token;
  compile_expression(expr);
  c = c2;
  m_bc_inst.RTN_FUNC(1);
  return(c);
}

/***********************************************************************
 * G__blockscope::compile_throw()
 *  throw;
 *  throw val ;
 *  throw(val);
 *        ^--> ^
 ***********************************************************************/
int G__blockscope::compile_throw(string& token,int c) {
  stdclear(token);
  switch(c) {
  case ';':
    // re-throw former exception which is already in stack.
    break;
  case '(':
    m_preader->putback();
    // Now that we put it back, let's process it
  case ' ':
  default:
    c = m_preader->fgetstream(token,";");

    // evaluate and instantiate exception object
    m_bc_inst.LD(0);

    // Set flag to generate ALLOCEXCEPTION instead of 
    // ALLOCTEMP/SETTEMP,STORETEMP ...  POPTEMP 
    G__throwingexception = 1;
    compile_expression(token); 
    G__throwingexception = 0;

    break;
  }
  m_bc_inst.THROW(); // THROW instruction stores exception 
                     // object in G__exceptionbuffer
  return(c);
}

/***********************************************************************
 * G__blockscope::compile_catch()
 *  catch(type x) {   }
 *  catch(...) {   }
 *  other statement
 * ^
 ***********************************************************************/
int G__blockscope::compile_catch(string& token,int c) {
  m_preader->storepos();
  stdclear(token);

  c = m_preader->fgettoken(token);

  if(token=="catch" && '('==c) {
    //  catch(type x) {   }
    //  catch(type) {   }
    //  catch(...) {   }
    //        ^
    stdclear(token);
    c = m_preader->fgettoken(token);
    if(""==token && '.'==c) {
      c = m_preader->fignorestream(")");
      //  catch(...) {   }
      //            ^
      G__blockscope catchblock(this);
      c = catchblock.compile();
      m_bc_inst.DESTROYEXCEPTION();
      return(0);
    }

    G__TypeReader type;
    while(type.append(token,c)) {
      c = m_preader->fgettoken(token);
    }

    G__value typevalue = type.GetValue();
    m_bc_inst.TYPEMATCH(&typevalue);

    int pnext_catchblock = m_bc_inst.CNDJMP();

    G__blockscope catchblock(this);
    catchblock.m_bc_inst.ENTERSCOPE();

    if(token=="" && ')'==c) {
      // catch(type) { }
      //            ^
    }
    else {
      // catch(type x) { }
      //              ^
      // allocate variable 
      int ig15=0; // dummy
      struct G__var_array *var;
      deque<int> arysize; // dummy
      deque<int> typesize; // dummy
      var = catchblock.allocatevariable(type,token,ig15,arysize,typesize,0);
      if(type.Isreference()) catchblock.m_bc_inst.INIT_REF(var,ig15,0,'p');
      else                   catchblock.m_bc_inst.ST_LVAR(var,ig15,0,'p');
      //m_bc_inst.POP();
    }

    c = catchblock.compile_core();

    catchblock.m_bc_inst.EXITSCOPE();

    m_bc_inst.DESTROYEXCEPTION();

    int pendof_catchblock = m_bc_inst.JMP();

    m_bc_inst.Assign(pnext_catchblock,m_bc_inst.GetPC());

    return(pendof_catchblock); // jump destination resolved in compiled_try()
  }
  else {
    //  other statement without catch(...)
    m_preader->rewindpos();
    stdclear(token);
    m_bc_inst.THROW(); // re-throw 
    return(0);
  }

  return(c);
}

/***********************************************************************
 * G__blockscope::compile_try()
 *  try {   }
 *       ^
 ***********************************************************************/
int G__blockscope::compile_try(string& token,int c) {

  int pfirst_catchblock = m_bc_inst.TRY();
  int pendof_catchblock = pfirst_catchblock+1;
  G__breaktable jump_endof_catchblock;
  jump_endof_catchblock.add(pendof_catchblock);

  G__blockscope tryblock(this);
  tryblock.compile(1); // ENTERSCOPE, statements... ,EXITSCOPE,  c='}'
  //  try {   }
  //           ^
  m_bc_inst.RTN_FUNC(2);

  m_bc_inst.Assign(pfirst_catchblock,m_bc_inst.GetPC());

  int pnext_catchblock;
  for(;;) {
    pnext_catchblock = compile_catch(token,c);
    if(0==pnext_catchblock) {
       //  other statement  without catch(...) 
       // ^
       //  catch(...) {   }
       //                  ^
       break;
     }
     else {
       // catch(type x) { }
       // catch(type)   { }
       //                  ^
       jump_endof_catchblock.add(pnext_catchblock);
     }
  }

  jump_endof_catchblock.resolve(m_bc_inst,m_bc_inst.GetPC());
  c = '}';
  return(c);
}

////////////////////////////////////////////////////////////////////////////
// brace, 2nd level
////////////////////////////////////////////////////////////////////////////

/***********************************************************************
 * G__blockscope::compile_brace()
 *	do {
 *	try {
 *	union {   // anonymous union
 *      {
 ***********************************************************************/
int G__blockscope::compile_brace(string& token,int c) {
  if(token=="do")       c = compile_do(token,c);   // c==';'
  else if(token=="try") c = compile_try(token,c);  // c=='}'
  else if(token=="union") {
    stdclear(token);
    int store_type = G__struct.type[G__tagdefining];
    G__struct.type[G__tagdefining] = 'u';
    c = G__blockscope::compile_core(1);
    G__struct.type[G__tagdefining] = store_type;
    c = m_preader->fignorestream(";");
  }
  else if(token=="") {
    G__blockscope block(this);
    c = block.compile(1);                    // c=='}'
  }
  return(c);
}


////////////////////////////////////////////////////////////////////////////
// declaration, 2nd level
////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////
int G__blockscope::compile_declaration(G__TypeReader& type,string& token,int c) {
  //  type varname ;
  //  type varname = expr;
  //  type varname [] = { } ;
  //  type objname (arglist);
  //  type funcname(arglist);
  //                ^
  deque<int> arysize;
  deque<int> typesize;
  G__var_array *var;
  int ig15;
  int isextrapointer=0;

  if(token=="operator") {
    // type operator@@   ( arglist);
    //               ^ -> ^
    do {
      if(c && !isspace(c)) token.append((string::size_type)1,(char)c);
      c = m_preader->fgetc();
    } while(c!='(') ;
  }

  if('('==c) { // function or constructor
    if(Isfunction(token)) { // or should look for typename instead???
      // for the time being, function prototype within function can be ignored.
      // In future, this part should call G__make_ifunctable()
      c = m_preader->fignorestream(";");
      stdclear(token);
      return(c);
    }
    else if(token=="") {
      // type (*p)(args);
      // type (p)[2][3];
      // type (*p)[2][3];
      // type (*p[3][4])[2][3];
      //       ^
      c = readtypesize(token,typesize,isextrapointer); // c== ';' ',' '['
    }
    else {
      // type name (arglist);
      //            ^
      var = allocatevariable(type,token,ig15,arysize,typesize,0);
      if(type.Property()&G__BIT_ISREFERENCE) {
        // type& name (expr);
        //             ^ --> ^
	c = init_reftype(token,var,ig15,c);
        //  c== ';' ','
      }
      else if(type.Property()&(G__BIT_ISFUNDAMENTAL|G__BIT_ISPOINTER|G__BIT_ISENUM)){
        // type name (expr);
        //            ^ --> ^
        c = initscalar(type,var,ig15,token,c); // c== ';' ','
      }
      else if(type.Property()&(G__BIT_ISSTRUCT|G__BIT_ISCLASS)) {
        // type name (arglist);
        //            ^ ---->  ^
        c = init_w_ctor(type,var,ig15,token,c); // c== ';' or ','
      }
      else /* if(type.Property&(G__BIT_ISUNION)) */ {
        // error;
        G__fprinterr(G__serr,"Error: No constructor for union %s",type.Name());
        G__genericerror((char*)NULL);
      }
      // c== ';' ','
      goto l_nextiter;
    }
  }

  while('['==c) {
    // declaration of an array, read [2][3][4]
    // type p   [2][3] = { };
    // type (*p)[2][3] ;
    //           ^ ---> ^  c is either ';' ',' or '=';
    c = readarraysize(arysize); // c== '=' ';' ','
    // array of size 1 , work-around -> changed to size 2
    if(arysize.size()==1 && arysize[0]==1) arysize[0]=2;
  }

  // create an entry for the scope variable here, we should have var and ig15
  var = allocatevariable(type,token,ig15,arysize,typesize,isextrapointer);

  if((';'==c||','==c)&& (type.Property()&(G__BIT_ISSTRUCT|G__BIT_ISCLASS)) &&
     0==(type.Property()&(G__BIT_ISPOINTER|G__BIT_ISREFERENCE))) {
    // type  a; 
    //         ^
    c = init_w_defaultctor(type,var,ig15,token,c); // c==';' ','
  }

  if(
     type.Isstatic()
     ) {
    // this is a static object, 
    // get handle from global table and make a link
    // and then ignore initialization, if exists
    stdclear(token);
    if(','!=c && ';'!=c) c = m_preader->fignorestream(";,"); // c==';' ','
    goto l_nextiter;
  }

  if(c=='=') {
    // initialization if exists
    // this part generates bytecode, but it depends on constness and staticness
    // type a[] = { };
    // type a   = { };
    // type a   = expr;
    // type a   = func(args);
    // type a   = type(args);
    //           ^
    c = read_initialization(type,var,ig15,token,c); // c==';' ','
  }  

 l_nextiter: // c==';' ','
  stdclear(token);

  if(c==',') {
    type.nextdecl();
    do {
      c = m_preader->fgettoken(token);
    } while(type.append(token,c)) ; 
    c = compile_declaration(type,token,c); // recursive call
  }

  if(c!=';') {
    G__genericerror("Error: missing ';'");
  }

  return(c);
}

////////////////////////////////////////////////////////////////////////////
// object initialization
////////////////////////////////////////////////////////////////////////////

/***********************************************************************
 * G__blockscope::read_initialization()
 ***********************************************************************/
int G__blockscope::read_initialization(G__TypeReader& type
				       ,struct G__var_array* var,int ig15
				       ,string& token,int c) {

  size_t *varlabel = var->varlabel[ig15];

  stdclear(token);

  if(varlabel[0]==1 && varlabel[1]==0) {
    // single object initialization
    if(type.Property()&G__BIT_ISREFERENCE) {
      // type& x = val;
      //          ^
      c = init_reftype(token,var,ig15,c);
    }
    else if(type.Property()&(G__BIT_ISFUNDAMENTAL|G__BIT_ISPOINTER|G__BIT_ISENUM)){
      // type x = val;
      //         ^
      c = initscalar(type,var,ig15,token,c); // c== ';' or ','
    }
    else if(type.Property()&(G__BIT_ISSTRUCT|G__BIT_ISCLASS)) {
      // type x = { a, b, c};
      // type x = type(arg);   -> type::ctor(arg);
      // type x = val;         -> type::ctor(valtype&); 
      //         ^
      c = m_preader->fgetstream_template(token,"{(;" /* ,0 */ );
      if(c=='{' && token=="") {
        c = initstruct(type,var,ig15,token,c); // c== ';' or ','
      }
      else if(c=='(') {
        G__TypeReader itype;
        if(itype.append(token,0) && type==itype) {
          // type x  = type(arg);  -> ctor
          //                ^
          c = init_w_ctor(type,var,ig15,token,c); // c== ';' or ','
        }
        else {
          // type x  = func(arg);
          //                ^
          token.append((string::size_type)1,(char)c);
          string autre;
          c = m_preader->fgetstream(autre,";,",1); // c== ';' or ','
          token.append(autre);
          // type x  = func(arg);
          //                     ^
          c = init_w_expr(type,var,ig15,token,c); // c== ';' ','
        }
      }
      else /* c==';' */ {
        // type x  = expr;
        //                ^
        c = init_w_expr(type,var,ig15,token,c); // c== ';' ','
      }
    }
    else /* if(type.Property&(G__BIT_ISUNION)) */ {
      // error;
      G__fprinterr(G__serr,"Error: No constructor for union %s",type.Name());
      G__genericerror((char*)NULL);
    }
  }
  else {
    // array initialization
    if(type.Property()&(G__BIT_ISFUNDAMENTAL|G__BIT_ISPOINTER|G__BIT_ISENUM)){
      c = initscalarary(type,var,ig15,token,c); // c== ';' ','
    }
    else if(type.Property()&(G__BIT_ISSTRUCT|G__BIT_ISCLASS)) {
      if(G__struct.funcs[type.Tagnum()]&G__HAS_XCONSTRUCTOR) {
        // string a[] = { "abc" , "def" , "hij" };
        //               ^
        c = initstructary(type,var,ig15,token,c); // c== ';' ','
      }
      else {
        // A x   = { "abc" , 123, 3.45 };
        // A x[] = { {"abc",123,3.45},{"def",456,6.78} };
        //        ^
        c = m_preader->fgetstream_template(token,"{(;" /* ,0 */ );
        c = initstruct(type,var,ig15,token,c); // c== ';' ','
      }
    }
    else /* if(type.Property&(G__BIT_ISUNION)) */ {
      // error;
      G__fprinterr(G__serr,"Error: No constructor for union %s",type.Name());
      G__genericerror((char*)NULL);
    }
  }

  stdclear(token);

  return(c); // c== ';' ','
}

/***********************************************************************
 * G__blockscope::init_reftype()
 ***********************************************************************/
int G__blockscope::init_reftype(string& token
				,struct G__var_array* var,int ig15,int c) {
  // type& name ( expr );
  // type& x    = val   ;
  //             ^ -->   ^
  stdclear(token);
  c = m_preader->fgetstream(token,");,"  /* ,(c=='(')?1:0 */ );
  compile_expression(token);
  m_bc_inst.INIT_REF(var,ig15,0,'p');
  if(c==')') c = m_preader->fignorestream(";,");
  return(c); // c== ';' ','
}

/***********************************************************************
 * G__blockscope::init_w_ctor(),  class object only
 ***********************************************************************/
int G__blockscope::init_w_ctor(G__TypeReader& type
				,struct G__var_array* var,int ig15
				,string& token,int c) {
  // type x  = type(arg);  -> ctor
  // type x        (arg);  -> ctor
  //                ^                  , token=="type"||"x"  c=='('
  struct G__param* para = new G__param();
  para->paran=0;

  do {
    stdclear(token);
    c = m_preader->fgetstream(token,",)" /* ,1 */ );
    para->para[para->paran++] = compile_expression(token);
  } while(c==',');
  para->para[para->paran] = G__null;

  // type x  = type(arg);  -> ctor
  //                ^-> ^

  call_ctor(type,para,var,ig15,0);

  c = m_preader->fignorestream(";,");
  delete para;
  return(c); // c== ';' or ','
}

/***********************************************************************
 * G__blockscope::init_w_defaultctor(),  class object only
 ***********************************************************************/
int G__blockscope::init_w_defaultctor(G__TypeReader& type
				      ,struct G__var_array* var,int ig15
				      ,string& /*token*/,int c) {
  // type  name;   -> default ctor or nothing
  //            ^    token=="name"   c==';'
   struct G__param* para = new G__param();
  para->paran = 0;
  para->para[0] = G__null;

  int num = var->varlabel[ig15][1]; // get array size
  if (num > 0) {
    m_bc_inst.LD(num);
    m_bc_inst.SETARYINDEX(1);
    call_ctor(type,para,var,ig15,num);
    m_bc_inst.RESETARYINDEX(1);
  }
  else {
    call_ctor(type,para,var,ig15,0);
  }
  delete para;
  return(c); // c== ';' ',' no change
}

/***********************************************************************
 * G__blockscope::init_w_expr(),  class object only
 ***********************************************************************/
int G__blockscope::init_w_expr(G__TypeReader& type
				,struct G__var_array* var,int ig15
				,string& token,int c) {
  // type x  =      expr;    -> ctor , assignment operator is not allowed
  // type x  = func(arg);    -> ctor , assignment operator is not allowed
  //                     ^     token=="expr"   c==';'

  struct G__param* para = new G__param();
  para->paran = 1;
  para->para[0] = compile_expression(token);
  para->para[1] = G__null;

  call_ctor(type,para,var,ig15,0);

  delete para;
  return(c); // c== ';' ','
}

/***********************************************************************
 * G__blockscope::call_ctor
 ***********************************************************************/
int G__blockscope::call_ctor(G__TypeReader& type,struct G__param *libp
				,struct G__var_array* var,int ig15,int num) {

  // GetMethod finds a function with type conversion, however,
  // bytecode for type conversion is not generated. There must be similar
  // situation in other place in this source file.
  long dmy;
  G__MethodInfo m = type.GetMethod(type.TrueName(),libp,&dmy
				   ,G__ClassInfo::ConversionMatchBytecode
				   );

  if(m.IsValid()) {
    if(!access(m)) {
      G__fprinterr(G__serr,"Error: function '%s' is private or protected"
		   ,m.Name());
      G__genericerror((char*)NULL);
      return(0);
    }
    struct G__ifunc_table *ifunc = (struct G__ifunc_table*)m.Handle();
    int ifn = m.Index();
    if(type.Property()&G__BIT_ISCOMPILED) {
      // This is for compiled class
      m_bc_inst.CTOR_SETGVP(var,ig15,0); // init local block scope object
      m_bc_inst.LD_FUNC_BC(ifunc,ifn,libp->paran,(void*)m.InterfaceMethod());
      m_bc_inst.SETGVP(-1); // restoration from store_globalvarpointer stack
    }
    else {
      // This is for interpreted class
      m_bc_inst.LD_LVAR(var,ig15,0,'p');
      m_bc_inst.PUSHSTROS();
      m_bc_inst.SETSTROS();
      Baseclassctor_vbase(var->p_tagtable[ig15]);
    if(num)
      m_bc_inst.LD_FUNC_BC(ifunc,ifn,0,(void*)G__bc_exec_ctorary_bytecode);
    else 
      m_bc_inst.LD_FUNC_BC(ifunc,ifn,libp->paran,(void*)G__bc_exec_ctor_bytecode);
    //m_bc_inst.LD_FUNC_BC(ifunc,ifn,libp->paran,(void*)G__bc_exec_normal_bytecode);
      m_bc_inst.POPSTROS();
    }
    return(1);
  }
  else {
    // if paran==0 || (paran==1 && para[0].type==type fine
    // otherwise, error
    G__fprinterr(G__serr,"Error: '%s' has no such constructor",type.Name());
    G__genericerror((char*)NULL);
  }
  return(0);
}

/***********************************************************************
 * G__blockscope::call_func
 ***********************************************************************/
G__value G__blockscope::call_func(G__ClassInfo& cls
			     ,const string& fname,struct G__param *libp
			     ,int /*memfuncflag*/,int isarray
			     ,G__ClassInfo::MatchMode mode
			     ) {

  // GetMethod finds a function with type conversion, however,
  // bytecode for type conversion is not generated. There must be similar
  // situation in other place in this source file.
  long dmy;
  G__MethodInfo m = cls.GetMethod(fname.c_str(),libp,&dmy
				  ,mode // ConversionMatch ???
				  );
  if(m.IsValid()) {
    if(!access(m)) {
      G__fprinterr(G__serr,"Error: function '%s(",m.Name());
      G__MethodArgInfo arg(m);
      int stat=1;
      while(arg.Next()) {
        if(stat) { G__fprinterr(G__serr,","); stat=0; }
        G__fprinterr(G__serr,"%s %s",arg.Type()->Name(),arg.Name());
        if(arg.DefaultValue()) G__fprinterr(G__serr,"=%s",arg.DefaultValue());
      }
      G__fprinterr(G__serr,")' is private or protected");
      G__genericerror((char*)NULL);
      return(G__null);
    }
    struct G__ifunc_table *ifunc = (struct G__ifunc_table*)m.Handle();
    int ifn = m.Index();
    if(cls.Property()&G__BIT_ISCOMPILED) {
      // This is for compiled class
      m_bc_inst.LD_FUNC_BC(ifunc,ifn,libp->paran,(void*)m.InterfaceMethod());
    }
    else {
      // This is for interpreted class
      if(m.Property()&G__BIT_ISVIRTUAL) 
        m_bc_inst.LD_FUNC_VIRTUAL(ifunc,ifn,libp->paran
				,(void*)G__bc_exec_virtual_bytecode);
      else if(fname==cls.Name()) {
	if(isarray)
	  m_bc_inst.LD_FUNC_BC(ifunc,ifn,libp->paran
			       ,(void*)G__bc_exec_ctorary_bytecode);
	else
	  m_bc_inst.LD_FUNC_BC(ifunc,ifn,libp->paran
			       ,(void*)G__bc_exec_ctor_bytecode);
      }
      else {
	if(isarray) {
	  if('~'==fname[0]) 
	    m_bc_inst.LD_FUNC_BC(ifunc,ifn,libp->paran
				 ,(void*)G__bc_exec_dtorary_bytecode);
	  else // this must be operator=
	    m_bc_inst.LD_FUNC_BC(ifunc,ifn,libp->paran
				 ,(void*)G__bc_exec_ctorary_bytecode);
	}
	else {
	  m_bc_inst.LD_FUNC_BC(ifunc,ifn,libp->paran
			       ,(void*)G__bc_exec_normal_bytecode);
	}
      }
    }
    G__value result = m.Type()->Value();
    return(result);
  }
  return(G__null);
}

/***********************************************************************
 * G__blockscope::initscalar, fundamental type or pointer only
 ***********************************************************************/
int G__blockscope::initscalar(G__TypeReader& type
			     ,struct G__var_array* var,int ig15
			     ,string& token,int c) {
  c = m_preader->fgetstream(token,");," /* ,(c=='(')?1:0 */ );
  G__value result=compile_expression(token);
  G__TypeReader rtype(result);
  if(!G__Isvalidassignment(type,rtype,&result)) {
    G__fprinterr(G__serr,"Error: assignment type mismatch %s <= %s"
                  ,type.Name(),rtype.Name());
    G__genericerror((char*)NULL);
  }
  conversion(result,var,ig15,'p',0); // embed into G__Isvalidassignment ?
  m_bc_inst.ST_LVAR(var,ig15,0,'p');
  if(c==')') c = m_preader->fignorestream(";,");
  return(c); // c== ';' or ','
}

/***********************************************************************
 * G__blockscope::initstruct, struct and array of struct without ctor
 ***********************************************************************/
int G__blockscope::initstruct(G__TypeReader& type, struct G__var_array* var, int varid, string& /*token*/, int c)
{
  // MyClass x   = { "abc", 123, 3.45 };
  // MyClass x[] = { {"abc", 123, 3.45}, {"def", 456, 6.78} };
  //          ^
  // FIXME: We do not handle brace nesting properly,
  //        we need to default initialize members
  //        whose initializers were omitted.
  // We must be an aggregate type, enforce that.
  if (G__struct.baseclass[var->p_tagtable[varid]]->basen) {
    // -- We have base classes, i.e., we are not an aggregate.
    // FIXME: This test should be stronger, the accessibility
    //        of the data members should be tested for example.
    G__fprinterr(G__serr, "Error: %s must be initialized by constructor", type.Name());
    G__genericerror(0);
    int c1 = G__fignorestream("}");
    //  type var1[N] = { 0, 1, 2.. }  , ... ;
    // came to                      ^
    c1 = G__fignorestream(",;");
    //  type var1[N] = { 0, 1, 2.. } , ... ;
    // came to                        ^  or ^
    return c1;
  }
  int number_of_dimensions = var->paran[varid];
  size_t& num_of_elements = var->varlabel[varid][1];
  const int& stride = var->varlabel[varid][0];
  // Check for an unspecified length array.
  int isauto = 0;
  if (num_of_elements == INT_MAX /* unspecified length flag */) {
    // -- We are an unspecified length array.
    // a[] or  a[][B][C]
    // Set isauto flag and reset number of elements.
    isauto = 1;
    num_of_elements = 0;
  }
  // Load first address of array as pointer.
  for (int i = 0; i < number_of_dimensions; ++i) {
    m_bc_inst.LD(0);
  }
  m_bc_inst.LD_LVAR(var, varid, number_of_dimensions, 'P');
  // Initialize buf.
  G__value buf;
  buf.type = toupper(var->type[varid]);
  buf.tagnum = var->p_tagtable[varid];
  buf.typenum = var->p_typetable[varid];
  buf.ref = 0;
  buf.obj.reftype.reftype = var->reftype[varid];
  // Get size.
  int size = 0;
  if (islower(var->type[varid])) {
    // -- We are *not* a pointer.
    size = G__sizeof(&buf);
  }
  else {
    // -- We are a pointer, handle as a long.
    buf.type = 'L';
    size = G__LONGALLOC;
  }
  G__ASSERT((stride > 0) && (size > 0));
  // Get a pointer to the first data member.
  int memindex = 0;
  struct G__var_array* memvar = G__initmemvar(var->p_tagtable[varid], &memindex, &buf);
  //
  // Read and process the initializer specification.
  //
  int mparen = 1;
  long linear_index = -1;
  buf.obj.i = var->p[varid] + memvar->p[memindex];
  G__FastAllocString expr(G__ONELINE);
  while (mparen) {
    // -- Read the next initializer value.
    int c1 = G__fgetstream(expr, 0, ",{}");
    if (expr[0]) {
      // -- We have an initializer expression.
      // FIXME: Do we handle a string literal correctly here?
      ++linear_index;
      // If we are an array, make sure we have not gone beyond the end.
      if ((num_of_elements || isauto) && (linear_index >= (long)num_of_elements)) {
        // -- We have gone past the end of the array.
	if (isauto) {
          // -- Unspecified length array, make it bigger to fit.
          // Allocate another stride worth of elements.
          num_of_elements += stride;
	}
	else {
          // -- Fixed-size array, error, array index out of range.
	  G__fprinterr(G__serr, "Error: %s: %d: Array initialization out of range *(%s+%ld), upto %lu ", __FILE__, __LINE__, type.Name(), linear_index, num_of_elements);
	  G__genericerror(0);
          while (mparen-- && (c1 != ';')) {
            c1 = G__fignorestream("};");
          }
          if (c1 != ';') {
            c1 = G__fignorestream(";");
          }
          return c1;
	}
      }
      // Loop over the data members and initialize them.
      G__TypeReader type_tmp;
      do {
        // FIXME: Do we have possible overflow problems here?
        int offset = ((var->p[varid] + (linear_index * size)) + memvar->p[memindex]) - buf.obj.i;
        buf.obj.i += offset;
        m_bc_inst.LD(offset);
        m_bc_inst.OP2(G__OPR_ADDVOIDPTR);
        type_tmp.Init(memvar, memindex);
        type_tmp.incplevel();
        m_bc_inst.CAST(type_tmp);
        /* G__value reg = */ G__getexpr(expr);
        m_bc_inst.LETNEWVAL();
        // Move to next data member.
	memvar = G__incmemvar(memvar, &memindex, &buf);
        if ((c1 == '}') || !memvar) {
          // -- All done if no more data members, or end of list.
          // FIXME: We are not handling nesting of braces properly.
          //        We need to default initialize the rest of the members.
          break;
        }
        // Get next initializer expression.
        c1 = G__fgetstream(expr, 0, ",{}");
      } while (memvar);
      // Reset back to the beginning of the data member list.
      memvar = G__initmemvar(var->p_tagtable[varid], &memindex, &buf);
    }
    // Change parser state for next initializer expression.
    switch (c1) {
      case '{':
        // -- Increment nesting level.
        ++mparen;
        break;
      case '}':
        // -- Decrement nesting level and move to next dimension.
        --mparen;
        break;
      case ',':
        // -- Normal end of an initializer expression.
        break;
    }
  }
  if (isauto) {
    // -- An unspecified length array.
    var->p[varid] = G__malloc(num_of_elements, size, var->varnamebuf[varid]);
  }
  // Read and discard up to next comma or semicolon.
  c = G__fignorestream(",;");
  // MyClass var1[N] = { 0, 1, 2.. } , ... ;
  // came to                        ^  or ^   */
  //
  // Note: The return value c is either a comma or a semicolon.
  return c;
}

/***********************************************************************
 * G__blockscope::initscalarary, fundamental type or pointer array only
 ***********************************************************************/
int G__blockscope::initscalarary(G__TypeReader& /*type*/, struct G__var_array* var, int ig15, string& /*token*/, int c)
{
  // char* ary[]  = { "a", "b" }; 
  // char* ary[n] = { "a", "b" }; 
  // char  ary[]  = "abc"; 
  // char  ary[4] = "abc"; 
  // char  ary[3] = "abc"; // ary[4]=0; +1 element is allocated in allocvar
  // int   ary[]  = { 1,2,3 };
  // int   ary[n] = { 1,2,3 };
  //               ^
  G__FastAllocString expr(G__ONELINE);
  size_t& num_of_elements = var->varlabel[ig15][1];
  const int& stride = var->varlabel[ig15][0];
  // Check for an unspecified length array declaration.
  int isauto = 0;
  if (num_of_elements == INT_MAX /* unspecified length flag */) {
    // Set isauto flag and reset num_of_elements.
    isauto = 1;
    num_of_elements = 0;
  }
  // Load the address of the first element of the array as a pointer.
  const short num_of_dimensions = var->paran[ig15];
  for (int j = 0; j < num_of_dimensions; ++j) {
    m_bc_inst.LD(0);
  }
  m_bc_inst.LD_LVAR(var, ig15, num_of_dimensions, 'P');
  // initialize buf
  G__value buf;
  buf.type = toupper(var->type[ig15]);
  buf.tagnum = var->p_tagtable[ig15];
  buf.typenum = var->p_typetable[ig15];
  buf.ref = 0;
  buf.obj.reftype.reftype = var->reftype[ig15];
  // Get the size of an element of the array.
  int size = 0;
  int typedefary = 0; 
  if (islower(var->type[ig15])) {
    if ((buf.typenum != -1) && G__newtype.nindex[buf.typenum]) {
      char store_var_type = G__var_type;
      size = G__Lsizeof(G__newtype.name[buf.typenum]);
      G__var_type = store_var_type;
      typedefary = 1; 
    }
    else {
      size = G__sizeof(&buf);
    }
  }
  else {
    // pointer assignment handled as long
    buf.type = 'L';
    size = G__LONGALLOC;
  }
  if ((stride < 0) || (size <= 0)) {
    G__genericerror("Error: cint internal error");
  }
  //
  // Read initialization list.
  //
  c = G__fgetstream(expr, 0, ",;{}");
  if (c == ';') {
    // -- Should be a one-dimensional character array.
    // char  ary[] =  "abc";  
    //                     ^
    if ((var->type[ig15] != 'c') || (var->paran[ig15] != 1)) {
      G__fprinterr(G__serr, "Error: %s: %d: illegal initialization of '%s'", __FILE__, __LINE__, var->varnamebuf[ig15]);
      G__genericerror(0);
    }
    m_bc_inst.LD(0);
    m_bc_inst.LD_LVAR(var, ig15, 1, 'p');
    G__value reg = G__getexpr(expr);
    conversion(reg, var, ig15, 'p', 0);
    m_bc_inst.LETNEWVAL();
    if (num_of_elements == INT_MAX /* unspecified length flag */) {
      num_of_elements = strlen((char*) reg.obj.i) + 1;
    }
    return c;
  }
  if (c != '{') {
    G__genericerror("Error: syntax error, array initialization");
  }
  int mparen = 1;
  int inc = 0;
  int pi = num_of_dimensions;
  size_t linear_index = 0;
  int prev = 0;
  int stringflag = 0;
   while (mparen) {
      // -- Get next initializer expression.
      c = G__fgetstream(expr, 0, ",{}");
      if (expr[0]) {
         // -- We got an initializer expression.
         if ((var->type[ig15] == 'c') && (expr[0] == '"')) {
            // -- Character array initialized by a string literal.
            if (!typedefary) {
               size = var->varlabel[ig15][var->paran[ig15]];
            }
            stringflag = 1;
            if ((size < 0) && !num_of_elements) {
               isauto = 0;
               size = 1;
               stringflag = 2;
            }
         }
         prev = linear_index;
         if (inc) {
            linear_index = (linear_index - (linear_index % inc)) + inc;
         }
         if ((num_of_elements || isauto) && (linear_index >= num_of_elements)) {
            if (isauto) {
               num_of_elements += stride;
            }
            else if (stringflag == 2) {
            }
            else {
               // Error, array index out of range.
               G__fprinterr(G__serr, "Error: %s: %d: Array initialization over-run '%s'", __FILE__, __LINE__, var->varnamebuf[ig15]);
               G__genericerror(0);
               return c;
            }
         }
         // Default initialize omitted elements.
         for (size_t i = prev + 1; i < linear_index; ++i) {
            m_bc_inst.LD(&G__null);
            m_bc_inst.LETNEWVAL();
            m_bc_inst.OP1(G__OPR_PREFIXINC);
         }
         // Initialize this element.
         G__value reg;
         {
            int store_prerun = G__prerun;
            G__prerun = 0;
            // todo, only if !stringflag 
            reg = G__getexpr(expr);
            G__prerun = store_prerun;
            conversion(reg, var, ig15, 'p', 0);
         }
         if (stringflag == 1) {
         }
         else if ((stringflag == 2) && isauto) {
            num_of_elements = std::strlen((char*) reg.obj.i) + 1;
         }
         else {
            m_bc_inst.LETNEWVAL();
            m_bc_inst.OP1(G__OPR_PREFIXINC);
         }
      }
      // Change parser state for the next initializer expression.
    switch (c) {
      case '{':
        // -- Increment nesting level.
        ++mparen;
        if (stringflag && (var->paran[ig15] > 2)) {
          inc *= var->varlabel[ig15][--pi];
        }
        else {
          inc *= var->varlabel[ig15][pi--];
        }
        break;
      case '}':
        // -- Decrement nesting level.
        --mparen;
        ++pi;
        break;
      case ',':
        // -- Normal end of an initializer expression.
        // Flag that we move to next linear element.
        inc = 1;
        pi = num_of_dimensions;
        break;
    }
  }
  // Default initialize the remaining elements.
  if (!stringflag) {
    int initnum = num_of_elements;
    if ((buf.typenum != -1) && G__newtype.nindex[buf.typenum]) {
      // -- We are a typedef.
      initnum /= size;
    }
    for (int i = linear_index + 1; i < initnum; ++i) {
      m_bc_inst.LD(&G__null);
      m_bc_inst.LETNEWVAL();
      m_bc_inst.OP1(G__OPR_PREFIXINC);
    }
  }
  if (isauto && size > 0) {
    // -- Unspecified length array.
    // Allocate in order to increment memory pointer
    // now that we know the final size.
    var->p[ig15]=G__malloc(num_of_elements, size, var->varnamebuf[ig15]);
  }
  // Read and discard up to the next comma or semicolon.
  c = G__fignorestream(",;");
  //  type var1[N] = { 0, 1, 2.. } , ... ;
  // came to                        ^  or ^ 
  return c;
}

/***********************************************************************
 * G__blockscope::initstructary, array of struct with ctor 
 ***********************************************************************/
int G__blockscope::initstructary(G__TypeReader& type, struct G__var_array* /*var*/, int /*ig15*/, string& /*token*/, int c)
{
  // string a[] = { "abc" , "def" , "hij" };
  //               ^
  G__fprinterr(G__serr, "Error: Initialization by aggregate is not allowed with a class with explicitly defined constructor '%s'", type.Name());
  G__genericerror(0);
  c = G__fignorestream(";");
  return c;
}

/////////////////////////////////////////////////////////////////////////
// automatic variable allocation
/////////////////////////////////////////////////////////////////////////
struct G__var_array* G__blockscope::allocatevariable(G__TypeReader& type
						     ,const string& name
						     ,int& ig15
						     ,deque<int>& arysize
						     ,deque<int>& typesize
						     ,int isextrapointer) {
  struct G__var_array* var = m_var;

  if(!isalpha(name[0]) && name[0]!='_' && name[0]!='$') {
    G__fprinterr(G__serr,"Error: illegal variable name '%s'",name.c_str());
    G__genericerror((char*)NULL);
  }

  // traverse scope variables and check if there is name duplication
  while(true) {
    for(ig15=0;ig15<var->allvar;ig15++) {
      if(name==var->varnamebuf[ig15]) {
        // error duplicated variable name in scope
        G__fprinterr(G__serr,"Error: duplicate variable declaration '%s'",name.c_str());
        G__genericerror((char*)NULL);
      } 
    }
    if(var->next) var = var->next;
    else break; // keep the last value of var
  }

  // create a new entry for automatic variable
  if(var->allvar<G__MEMDEPTH) {
    ig15 = var->allvar++;
  }
  else {
    var->next = (struct G__var_array *)malloc(sizeof(struct G__var_array)) ;
    memset(var->next,0,sizeof(struct G__var_array));
    var->next->tagnum = var->tagnum;
    var = var->next;
    var->allvar = 1;
    ig15 = 0;
  }

  // set name
  size_t len = name.size()+1;
  var->varnamebuf[ig15] = (char*)malloc(len);
  strncpy(var->varnamebuf[ig15],name.c_str(), len);
  int hash,tmp;
  G__hash(name.c_str(),hash,tmp);
  var->hash[ig15] = hash;
  var->access[ig15] = G__PUBLIC;

  // set array size
  setarraysize(type,var,ig15,arysize,typesize,isextrapointer);

  // set type
  var->p_typetable[ig15] = type.Typenum();
  var->p_tagtable[ig15] = (short)type.Tagnum();
  if(type.Isreference()) {
    switch(type.Ispointer()) {
    case 0: 
      var->type[ig15] = tolower(type.Type());
      var->reftype[ig15] = G__PARAREFERENCE; 
      break;
    case 1:
      var->type[ig15] = toupper(type.Type());
      var->reftype[ig15] = G__PARAREFERENCE; 
      break;
    default:
      var->type[ig15] = toupper(type.Type());
      var->reftype[ig15] = G__PARAREF + type.Ispointer();
      break;
    }
  }
  else {
    switch(type.Ispointer()) {
    case 0: 
      var->type[ig15] = tolower(type.Type());
      var->reftype[ig15] = G__PARANORMAL; 
      break;
    case 1:
      var->type[ig15] = toupper(type.Type());
      var->reftype[ig15] = G__PARANORMAL; 
      break;
    default:
      var->type[ig15] = toupper(type.Type());
      var->reftype[ig15] = type.Ispointer();
      break;
    }
  }

  // set property and resolve address or offset
  if(type.Isstatic()) {
    // todo, This implementation is not exactly correct in a sense that
    // static objects do not have block scope.
    var->statictype[ig15] = G__LOCALSTATIC;
    var->p[ig15] = getstaticobject(name,m_ifunc,m_iexist);
  }
  else {
    if(type.Isconst() && (type.Property()&G__BIT_ISFUNDAMENTAL)
       && 0==(type.Property()&G__BIT_ISPOINTER)) {
      var->statictype[ig15] = G__LOCALSTATIC;
      var->p[ig15] = getstaticobject(name,m_ifunc,m_iexist,1);
      if(var->p[ig15]) return(var);
    }
    var->statictype[ig15] = G__AUTO; 
    // this part corresponds to G__compiler::Init(),
    // the bytecode compiler uses legacy G__malloc() and it uses the last
    // entry of class/struct table as size/offset calculation buffer.
    int num = var->varlabel[ig15][1] /* number of elements */;
    if (num == INT_MAX) {
      num = 0;
    } else if (!num) {
      num = 1;
    }
    else {
      if (type.Type() == 'c') {
        // allow: ary[3] = "abc";
        num += 1;
      }
    }

    char *buf = (char*)malloc(name.size()+1);
    strcpy(buf,name.c_str()); // Okay we allocated enough memory

    int size = type.Size();
    // todo, Reference argument works with and without following line.
    //  Need to review what is going on...
    if(type.Isreference()) size = G__LONGALLOC;

    var->p[ig15]=G__malloc(num,size,buf); // legacy
    free((void*)buf);

    if((type.Property()&(G__BIT_ISSTRUCT|G__BIT_ISCLASS)) &&
       (0==(type.Property()&(G__BIT_ISPOINTER|G__BIT_ISREFERENCE)))) {
      m_bc_inst.PUTAUTOOBJ(var,ig15);
    }
  }

  return(var);
}

/////////////////////////////////////////////////////////////////////////
// array size handling
////////////////////////////////////////////////////////////////////////////
int G__blockscope::readarraysize(deque<int>& arysize) {
  //  type varname [ ] = { } ;
  //  type varname [3][2] = { } ;
  //  type varname [][3];
  //                ^
  string expr;
  int c;

  do {
    c = m_preader->fgetstream(expr,"]=;,"  /* ,1 */ );  // "]"

    if(expr=="") {
      arysize.push_back(INT_MAX);      
    }
    else {
      int sz = getstaticvalue(expr); 
      arysize.push_back(sz);
    }

    c = m_preader->fgetstream(expr,"[=;," /* ,0 */ ); // or should it be ")[=;,"
  } while(c=='[');

  return(c); // c== '=' ';' ',' ')'
}

////////////////////////////////////////////////////////////////////////////
int G__blockscope::readtypesize(string& token
				       ,deque<int>& typesize
				       ,int& isextrapointer) {
  // type (*p)(args);
  // type (p)[2][3];
  // type (*p)[2][3];
  // type (*p[3][4])[2][3];
  //       ^
  string expr;
  int c=0;

  // type (*p)(args);
  // type (*p)[2][3];
  // type (*p[3][4])[2][3];
  //       ^  c=='('
  c = m_preader->fgettoken(token);

  if(token=="") {
    if(c=='*') {
      ++isextrapointer;
      c = m_preader->fgettoken(token);
    }
    else if(c==')') {
      // error
      G__fprinterr(G__serr,"Syntax error");
      G__genericerror((char*)NULL);
    }
    else {
      // error
      G__fprinterr(G__serr,"Syntax error");
      G__genericerror((char*)NULL);
    }
  }

  if(c=='[') {
    // type (*p[3][4])[2][3];
    //          ^
    c=readarraysize(typesize); // c== ')' ,  but not '=' ';' ','
    // type (*p[3][4])[2][3];
    //                ^ c==')'
    string dmy;
    c = m_preader->fgettoken(dmy);
    // type (*p[3][4])[2][3];
    // type (*p[3][4]);
    //                 ^ c==';' ',' or '['. '[' is read in one upper level
  }
  else if(c==')') {
    // this is normal, token has varname and c is ')'
    string dmy;
    c = m_preader->fgettoken(dmy);
    // type (*p)[2][3];
    // type ( p);
    //           ^ c==';' ',' or '['. '[' is read in one upper level
  }
  else {
    // error;
    G__fprinterr(G__serr,"Syntax error");
    G__genericerror((char*)NULL);
  }

  return(c); // c== ';' ',' '['
}


////////////////////////////////////////////////////////////////////////////
/************************************************************
 * type array[A][B][C][D]
 *   var->varlabel[var_identity][0]=B*C*D; or 1;
 *   var->varlabel[var_identity][1]=A*B*C*D;
 *   var->varlabel[var_identity][2]=B;
 *   var->varlabel[var_identity][3]=C;
 *   var->varlabel[var_identity][4]=D;
 *   var->varlabel[var_identity][5]=1;
 * if type (*pary[A][B][C][D])[x][y][z]  A,B,C,D->typesize, x,y,z->arysize
 *   var->varlabel[var_identity][6]=x*y*z or 1;
 *   var->varlabel[var_identity][7]=x;
 *   var->varlabel[var_identity][8]=y;
 *   var->varlabel[var_identity][9]=z;
 *   var->varlabel[var_identity][10]=1;
 *   var->varlabel[var_identity][11]=0;
 ***********************************************************/
template<class T, class E>
void G__appendx(T& a, E& b)
{
  std::deque<int>::iterator first = a.begin();
  std::deque<int>::iterator last = a.end();
  for (; first != last; ++first) {
    b.push_back(*first);
  }
}

void G__blockscope::setarraysize(G__TypeReader& type, struct G__var_array* var, int ig15, std::deque<int>& arysize, std::deque<int>& typesize, int isextrapointer)
{

  // todo, This check criteria is not quite right. Need to review
  if (
    (
      !typesize.size() &&
      (arysize.size() >= (G__MAXVARDIM - 1))
    ) ||
    (
      !arysize.size() &&
      (typesize.size() >= (G__MAXVARDIM - 2))
    ) ||
    (
      typesize.size() &&
      arysize.size() &&
      ((arysize.size() + typesize.size()) > (G__MAXVARDIM - 3))
    )
  ) {
    G__fprinterr(G__serr, "Limitation: Cint can handle only up to %d dimension array", G__MAXVARDIM - 1);
    G__genericerror(0);
  }

  // todo, copy algorithm does not work. Maybe g++ bug -> implement alternative

  int flag = 0;
  std::deque<int> asize;

  // type (*ary)[x][y][z];
  // type (*ary[A][B][C]);
  // type (*ary[A][B][C])[x][y][z];
  if (isextrapointer) {
    type.incplevel();
  }

  if (!isextrapointer || !arysize.size() || !typesize.size()) {
    // type (ary[A][B][C])[x][y][z]; -> type ary[A][B][C][x][y][z];
    // type (*ary)[x][y][z];         -> type *ary[x][y][z];
    // type (*ary[A][B][C]);         -> type *ary[A][B][C];
    // type *ary[A][B][C];           -> type *ary[A][B][C];
    // type ary[A][B][C];            -> type ary[A][B][C];
    if (!arysize.size()) {
      G__appendx(typesize, asize);
      //copy(typesize.begin(),typesize.end(),back_inserter(asize));
    }
    else {
      G__appendx(arysize, asize);
      //copy(arysize.begin(),typesize.end(),back_inserter(asize));
    }
    //merge(typesize.begin(),typesize.end(),arysize.begin(),arysize.end()
    //      ,back_inserter(asize)); // VC++7.2 can not find merge algorithm
  }
  else {
    // type (*ary[A][B][C])[x][y][z];
    G__appendx(typesize, asize);
    //copy(typesize.begin(),typesize.end() ,back_inserter(asize));
    flag = 1;
  }

  var->paran[ig15] = arysize.size();
 
  if (!asize.size()) {
    // type a;
    var->varlabel[ig15][0] = 1;
    var->varlabel[ig15][1] = 0;
  }
  else {
    // type array[A][B][C][D]
    //  var->varlabel[var_identity][0]=B*C*D; or 1;
    //  var->varlabel[var_identity][1]=A*B*C*D;
    //  var->varlabel[var_identity][2]=B;
    //  var->varlabel[var_identity][3]=C;
    //  var->varlabel[var_identity][4]=D;
    //  var->varlabel[var_identity][5]=1;
    int stride = 1;
    int num_of_elements = 1;
    unsigned int i;
    for (i = 0; i < asize.size(); ++i) {
      num_of_elements *= asize[i];
      if (i) {
        stride *= asize[i];
        var->varlabel[ig15][i+1] = asize[i];
      }
    }
    var->varlabel[ig15][0] = stride;
    var->varlabel[ig15][i+1] = 1;
    // need more consideration for a[][2][3]
    if (asize[0] == INT_MAX) {
      var->varlabel[ig15][1] /* num of elements */ = INT_MAX /* unspecified length flag */;
    }
    else {
      var->varlabel[ig15][1] /* num of elements */ = num_of_elements;
    }
  }

  if (flag) {
    // if type (*pary[A][B][C][D])[x][y][z]  A,B,C,D->typesize, x,y,z->arysize
    //  var->varlabel[var_identity][6]=x*y*z or 1;
    //  var->varlabel[var_identity][7]=x;
    //  var->varlabel[var_identity][8]=y;
    //  var->varlabel[var_identity][9]=z;
    //  var->varlabel[var_identity][10]=1;
    //  var->varlabel[var_identity][11]=0;
    int a6 = asize.size() + 2;
    int a = 1;
    unsigned int i;
    for (i = 0; i <arysize.size(); ++i) {
      a *= arysize[i];
      var->varlabel[ig15][a6+1+i] = arysize[i];
    }
    var->varlabel[ig15][a6+1+i] = 1;
    var->varlabel[ig15][a6+2+i] = 0;
    var->varlabel[ig15][a6] = a;
  }

}


////////////////////////////////////////////////////////////////////////////
// preprocessor, 2nd level
////////////////////////////////////////////////////////////////////////////
/***********************************************************************
 * G__blockscope::compile_preprocessor()
 *    xxx##yyy
 *    # if
 *    # ifdef
 *    # ifndef
 *    # else
 *    # elif
 *    # endif
 *    # define
 *    # pragma
 *    # line
 *    # error
 ***********************************************************************/
//  THIS FUNCTION IS NOT NEEDED. This feature is implemented in G__reader
int G__blockscope::compile_preprocessor(string& token,int c) {
  if(token=="") {
    G__pp_command(); // legacy
    c = 0;
  }
  else {
    c = m_preader->fgetc();
    if(c=='#') {
      // todo token concatination
      //   AAA ## BBB
      //        ^
    }
    else {
      // todo  XXX  
      //       # preprocessor-command
      //        ^
    }
  }
 return(c);
}

//////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////
// name look up
////////////////////////////////////////////////////////////////////////////
/***********************************************************************
 * G__blockscope::Istypename()
 ***********************************************************************/
int G__blockscope::Istypename(const string& name) {
  size_t len = name.size();
  char *buf = new char[len+1];
  strncpy(buf,name.c_str(), len + 1);
  if(len>G__MAXNAME) {
    G__fprinterr(G__serr,"Limitation: Symbol name is too long %d>%d %s "
	         ,len,G__MAXNAME,buf);
    G__genericerror((char*)NULL);
  }
  int result=G__istypename(buf); // legacy
  delete [] buf;
 return(result);
}

/***********************************************************************
 * G__blockscope::Isfunction()
 ***********************************************************************/
int G__blockscope::Isfunction(const string& /*name*/) {
  // need to come back to think if this is a good way
  // always false for the time being.
  return(0);
}

////////////////////////////////////////////////////////////////////////////
// get address of static object in a function
////////////////////////////////////////////////////////////////////////////
long G__blockscope::getstaticobject(const string& varname
				    ,struct G__ifunc_table* ifunc,int ifn
				    ,int noerror) {
  // todo, This implementation is not exactly correct in a sense that
  // static objects do not have block scope.
  G__FastAllocString temp(G__ONELINE);
  int hash,i;
  struct G__var_array *var;

  if(-1!=ifunc->tagnum) 
     temp.Format("%s\\%x\\%x\\%x",varname.c_str(),ifunc->page,ifn,ifunc->tagnum);
  else
     temp.Format("%s\\%x\\%x" ,varname.c_str(),ifunc->page,ifn);

  G__hash(temp,hash,i)
  var = &G__global;
  do {
    i=0;
    while(i<var->allvar) {
      if((var->hash[i]==hash)&&(strcmp(var->varnamebuf[i],temp)==0)) {
	return(var->p[i]);
      }
      i++;
    }
    var = var->next;
  } while(var);

  if(!noerror) {
     G__fprinterr(G__serr,"Error: No memory for static object %s ",temp());
    G__genericerror((char*)NULL); //legacy
  }
  return(0);
}

//////////////////////////////////////////////////////////////////////////////
int G__blockscope::conversion(G__value& result,struct G__var_array* var
			      ,int ig15,int vartype,int paran) {
  if(0==baseconversion(result,var,ig15,vartype,paran)) {
    if(0==conversionopr(result,var,ig15,vartype,paran)) {
      return(0);
    }
  }
  return(1);
}
//////////////////////////////////////////////////////////////////////////////
int G__blockscope::baseconversion(G__value& result,struct G__var_array* var
				   ,int ig15,int /*vartype*/,int paran) {
  if(('U'==var->type[ig15] || ('u'==var->type[ig15]&&
                               G__PARAREFERENCE==var->reftype[ig15]))
     && var->type[ig15]==result.type
     && -1!=var->p_tagtable[ig15] && -1!=result.tagnum
     && var->p_tagtable[ig15]!=result.tagnum 
     && -1!=G__ispublicbase(var->p_tagtable[ig15],result.tagnum,(long)0)) {
    // Look for base class conversion (offset calc) only
    if(paran) G__bc_REWINDSTACK(paran);
    m_bc_inst.CAST(var->type[ig15],var->p_tagtable[ig15]
                   ,var->p_typetable[ig15]
                   ,(var->reftype[ig15]==G__PARAREFERENCE)?1:0);
    if(paran) G__bc_REWINDSTACK(-paran);
    result.tagnum = var->p_tagtable[ig15];
    return(1);
  }
  return(0);
}

//////////////////////////////////////////////////////////////////////////////
int G__blockscope::conversionopr(G__value& result,struct G__var_array* var
				 ,int ig15,int vartype,int paran) {
  if('u'==result.type) {
    // look for result_type::operator[target_type]();
    G__value target = G__null;
    target.type = var->type[ig15];
    target.tagnum = var->p_tagtable[ig15];
    target.typenum = -1;
    target.obj.reftype.reftype = var->reftype[ig15];
    target.isconst = 0;
    G__TypeReader target_type(target);
    switch(vartype) {
    case 'v': target_type.decplevel(); break;
    case 'P': target_type.incplevel(); break;
    case 'p': break;
    default: break;
    }
    string fname = "operator ";
    fname.append(target_type.Name());
    G__TypeReader ty(result);
    // This GetMethod() is fine for type convversion because there is no args
    long dmy;
    G__MethodInfo m = ty.GetMethod(fname.c_str(),"",&dmy);
    if(m.IsValid()) {
      if(paran) G__bc_REWINDSTACK(paran);
      // LD_LVAR , not needed, already done
      m_bc_inst.PUSHSTROS();
      m_bc_inst.SETSTROS();
      // LD_FUNC operator target_type()
      struct G__ifunc_table *ifunc = (struct G__ifunc_table*)m.Handle();
      int ifn = m.Index();
      if((ty.Property()&G__BIT_ISCOMPILED)) 
	m_bc_inst.LD_FUNC_BC(ifunc,ifn,0,(void*)m.InterfaceMethod());
      else 
	m_bc_inst.LD_FUNC_BC(ifunc,ifn,0,(void*)G__bc_exec_normal_bytecode);
      m_bc_inst.POPSTROS();
      if(paran) G__bc_REWINDSTACK(-paran);
      // ST_VAR , not needed, will be done after this
      result = target_type.Value();
      return(1);
    }
    return(0);
  }
  return(0);
  // ??? target_type::operator=(target_type,result_type); assignment opr
  // ??? target_type::target_type(result_type);  ctor
}
//////////////////////////////////////////////////////////////////////////////

/****************************************************************
* G__value G__blockscope::compile_newopr()
* 
* Called by:
*   G__getpower()
*
*      V
*  new type
*  new type[10]
*  new type(53)
*  new (arena)type
*
*  string arena, type, aryindex, args;
*
* SETMEMFUNCENV
*
* 0 G__NEWALLOC       G__store_struct_offset <- malloc
* 1 size     0 if arena
* 2 isclass&&array
* stack
* sp-2     <- arena
* sp-1     <- pinc (aryindex)
* sp
*
* + Constructor
*   - array 
*     SETARYINDEX(1)    G__cpp_aryindex <- stack, also to memory header
*     call constructor
*     RESETARYINDEX(1)  restore  G__cpp_aryindex
*   - with arguments
*     Evaluate constructor arguments
*     call constructor
*   
* SET_NEWALLOC   G__store_struct_offset -> G__asm_stack
*
* RECMEMFUNCENV
*
****************************************************************/
G__value G__blockscope::compile_newopr(const string& expression) {
  G__srcreader<G__sstream> stringreader;
  stringreader.Init(expression.c_str());
  string arena,type,aryindex,args;
  int c=0;

  if(expression[0]=='(') {
    //  new (arena)type
    //      ^^ ->  ^
    c = stringreader.fgetc();
    c = stringreader.fgetstream(arena,")");
  }

  //  new        type;
  //  new        type[10];
  //  new        type(53);
  //  new (arena)type;
  //             ^ -> ^
  c = stringreader.fgetstream_template(type,"[(;");
  
  if(';'!=c) {
    switch(c) {
    case '[':
      c = stringreader.fgetstream(aryindex,"]");
      break;
    case '(':
      c = stringreader.fgetstream(args,")");
      break;
    default:
      break;
    }
    if(';'!=c) c = stringreader.fignorestream("[(;");
  }

  ///////////////////////////////////////////////////
  // parsing end, begin bytecode generation
  ///////////////////////////////////////////////////
  G__TypeReader ty(type.c_str());
  //G__TypeInfo ty(type.c_str());

  m_bc_inst.SETMEMFUNCENV();
  G__param* para = new G__param();
  long dmy=0;
  int isarena = arena.size();
  int isaryindex = aryindex.size();
  int isargs = args.size();

  /////////////////////////////////////////////////////////////////
  // Compiled class 
  if((ty.Property()&(G__BIT_ISCLASS|G__BIT_ISSTRUCT)) &&
     0==ty.Ispointer() && (ty.Property()&G__BIT_ISCOMPILED)) {
    if(isarena) {
      compile_expression(arena);
      m_bc_inst.SETGVP(0);
    }
    if(isaryindex) {
      compile_expression(aryindex);
      m_bc_inst.SETARYINDEX(1);
    }
    if(isargs) {
      compile_arglist(args,para);
    }
    else {
      para->paran=0;
      para->para[0]=G__null;
    }

    // GetMethod finds a function with type conversion, however,
    // bytecode for type conversion is not generated. There must be similar
    // situation in other place in this source file.
    G__MethodInfo m = ty.GetMethod(type.c_str(),para,&dmy
				   ,G__ClassInfo::ConversionMatchBytecode
				   );
    if(m.IsValid()) { // always true ???
      if(!access(m)) {
        G__genericerror("Error: can not call private or protected function");
        delete para;
        return(G__null);
      }
      struct G__ifunc_table *ifunc = (struct G__ifunc_table*)m.Handle();
      int ifn = m.Index();
      m_bc_inst.LD_FUNC_BC(ifunc,ifn,para->paran,(void*)m.InterfaceMethod());
    }
    else {
      // in case ctor is not found, -> error
      G__fprinterr(G__serr
           ,"Error: %s, there is no accessible constructor for operator new"
                   ,ty.Name());
      G__genericerror((char*)NULL);
    }

    if(isaryindex) m_bc_inst.RESETARYINDEX(1);
    if(isarena) m_bc_inst.SETGVP(-1);
    //m_bc_inst.SET_NEWALLOC(ty.Tagnum(),toupper(ty.Type()));
    ty.incplevel();
    m_bc_inst.SET_NEWALLOC(ty);
  }

  /////////////////////////////////////////////////////////////////
  // Interpreted class 
  else if((ty.Property()&(G__BIT_ISCLASS|G__BIT_ISSTRUCT)) &&
	  0==ty.Ispointer() && 0==(ty.Property()&G__BIT_ISCOMPILED)) {
    if(isarena) {
      compile_expression(arena);
    }
    if(isaryindex) {
      compile_expression(aryindex);
      m_bc_inst.PUSHCPY();
      m_bc_inst.SETARYINDEX(1);
    }
    else {
      m_bc_inst.LD(1);
    }
    m_bc_inst.PUSHSTROS();
    if(isarena) m_bc_inst.NEWALLOC(0,isaryindex?1:0);
    else        m_bc_inst.NEWALLOC(ty.Size(),isaryindex?1:0);

    if(args.size()) {
      compile_arglist(args,para);
    }
    else {
      para->paran=0;
      para->para[0]=G__null;
    }

    // GetMethod finds a function with type conversion, however,
    // bytecode for type conversion is not generated. There must be similar
    // situation in other place in this source file.
    G__MethodInfo m = ty.GetMethod(ty.Name(),para,&dmy
				   ,G__ClassInfo::ConversionMatchBytecode
				   );
    if(m.IsValid()) {
      if(!access(m)) {
        G__genericerror("Error: can not call private or protected function");
        delete para;
        return(G__null);
      }
      struct G__ifunc_table *ifunc = (struct G__ifunc_table*)m.Handle();
      int ifn = m.Index();
      Baseclassctor_vbase(ty.Tagnum());
      if(isaryindex)
        m_bc_inst.LD_FUNC_BC(ifunc,ifn,0,(void*)G__bc_exec_ctorary_bytecode);
      else 
        m_bc_inst.LD_FUNC_BC(ifunc,ifn,para->paran ,(void*)G__bc_exec_ctor_bytecode);
      //m_bc_inst.LD_FUNC_BC(ifunc,ifn,para->paran ,(void*)G__bc_exec_normal_bytecode);
    }
    else {
#ifdef G__NEVER
      if(para->paran==0 && (ty.ClassProperty()&G__CLS_HASDEFAULTCTOR)) {
	// fine, default constructor
      } else 
#endif
      {
	// in case ctor is not found, -> error
	G__fprinterr(G__serr
	     ,"Error: %s, there is no accessible constructor for operator new"
		     ,ty.Name());
	G__genericerror((char*)NULL);
      }
    }

    if(isaryindex) m_bc_inst.RESETARYINDEX(1);
    //m_bc_inst.SET_NEWALLOC(ty.Tagnum(),toupper(ty.Type()));
    ty.incplevel();
    m_bc_inst.SET_NEWALLOC(ty);
    m_bc_inst.POPSTROS();
  }
 

  /////////////////////////////////////////////////////////////////
  // Fundamental type and pointer
  else { 
    if(isarena) compile_expression(arena); // arena
    if(isaryindex) compile_expression(aryindex); // pinc
    else m_bc_inst.LD(1); // pinc
    m_bc_inst.PUSHSTROS();
    if(isarena) m_bc_inst.NEWALLOC(0,0);
    else        m_bc_inst.NEWALLOC(ty.Size(),0);

    //m_bc_inst.SET_NEWALLOC(ty.Tagnum(),toupper(ty.Type()));
    ty.incplevel();
    m_bc_inst.SET_NEWALLOC(ty);

    if(isargs) {
      compile_arglist(args,para);
      if(para->paran!=1) {
        // error
      }
    }
    else {
      para->paran=0;
      para->para[0]=G__null;
      m_bc_inst.LD(0);
    }

    m_bc_inst.LETNEWVAL();
    m_bc_inst.POPSTROS();
  }
  /////////////////////////////////////////////////////////////////

  //ty.incplevel(); // has to return pointer type
  G__value result = ty.Value();
  
  delete para;
  return(result);
}


/****************************************************************
* G__value G__bc_new_operator()
****************************************************************/
extern "C" G__value G__bc_new_operator(const char *expression) {
  return(G__currentscope->compile_newopr(string(expression)));
}

//////////////////////////////////////////////////////////////////////////////

/****************************************************************
* G__value G__blockscope::compile_deleteopr()
* 
*  delete   expr;   isarray=0
*  delete[] expr;   isarray=1
*                ^  expression="expr"
*
*   eval expr
*
*   // DELALLOCTABLE
*
*   PUSHSTROS
*   SETSTROS
*
*   GETARYINDEX
*
*   call dtor
*     a. virtual
*     b. normal
*     c. array
*
*   RESETARYINDEX
*
*   DELETEFREE
*
*   POPSTROS
*
****************************************************************/
void G__blockscope::compile_deleteopr(string& expression,int isarray) {
  G__value ptr;

  ptr = compile_expression(expression);

  G__TypeReader ty(ptr);
  ty.decplevel();

  G__param* para = new G__param();
  para->paran=0;
  para->para[0]=G__null;
  long dmy=0;

  m_bc_inst.PUSHCPY();
  int skip = m_bc_inst.CNDJMP();

  m_bc_inst.PUSHSTROS();
  m_bc_inst.SETSTROS();

  /////////////////////////////////////////////////////////////////
  // Compiled class
  if((ty.Property()&(G__BIT_ISCLASS|G__BIT_ISSTRUCT)) &&
     0==ty.Ispointer() && (ty.Property()&G__BIT_ISCOMPILED)) {

    if(isarray) m_bc_inst.GETARYINDEX();

    string dtorname = "~";
    dtorname.append(ty.Name());
    // This is fine because dtor doesn't have parameter
    G__MethodInfo m = ty.GetMethod(dtorname.c_str(),para,&dmy);
    if(m.IsValid()) { // always true ???
      if(!access(m)) {
        G__genericerror("Error: can not call private or protected function");
        delete para;
        return;
      }
      struct G__ifunc_table *ifunc = (struct G__ifunc_table*)m.Handle();
      int ifn = m.Index();
      // virtual function resolution and free is all done in compiled dtor.
      m_bc_inst.LD_FUNC_BC(ifunc,ifn,0,(void*)m.InterfaceMethod());
    }
    else {
      // dtor not found for compiled class, means dtor is private
      // implicit dtor is always generaed in bc_vtbl.cxx.
      G__genericerror("Error: can not call private or protected function");
      delete para;
      return;
    }

    if(isarray) m_bc_inst.RESETARYINDEX(1);
  }

  /////////////////////////////////////////////////////////////////
  // Interpreted class 
  else if((ty.Property()&(G__BIT_ISCLASS|G__BIT_ISSTRUCT)) &&
	  0==ty.Ispointer() && 0==(ty.Property()&G__BIT_ISCOMPILED)) {

    if(isarray) m_bc_inst.GETARYINDEX();

    string dtorname = "~";
    dtorname.append(ty.Name());
    // This is fine because dtor doesn't have parameter
    G__MethodInfo m = ty.GetMethod(dtorname.c_str(),para,&dmy);
    if(m.IsValid()) {
      if(!access(m)) {
        G__genericerror("Error: can not call private or protected function");
        delete para;
        return;
      }
      struct G__ifunc_table *ifunc = (struct G__ifunc_table*)m.Handle();
      int ifn = m.Index();

      if(isarray) {
        m_bc_inst.LD_FUNC_BC(ifunc,ifn,0,(void*)G__bc_exec_dtorary_bytecode);
      }
      else if(m.Property()&G__BIT_ISVIRTUAL) {
        m_bc_inst.LD_FUNC_VIRTUAL(ifunc,ifn,0,(void*)G__bc_exec_virtual_bytecode);
	isarray = 2; // this forces DELETEFREE to add object offset 
      }
      else {
        m_bc_inst.LD_FUNC_BC(ifunc,ifn,0,(void*)G__bc_exec_normal_bytecode);
      }
    }

    if(isarray) m_bc_inst.RESETARYINDEX(1);

    m_bc_inst.DELETEFREE(isarray);
  }
 
  /////////////////////////////////////////////////////////////////
  // fundamental type
  else { 
    m_bc_inst.DELETEFREE(isarray);
  }

  m_bc_inst.POPSTROS();

  m_bc_inst.Assign(skip,m_bc_inst.GetPC());
  delete para;
}


/****************************************************************
* G__value G__bc_delete_operator()
****************************************************************/
extern "C" void G__bc_delete_operator(const char *expression,int isarray) {
  string expr(expression);
  G__currentscope->compile_deleteopr(expr,isarray);
}

/****************************************************************
* access control
****************************************************************/
int G__blockscope::access(/* const */ G__MethodInfo& x) const {
  if(access(x.MemberOf()->Tagnum(),x.Property())) return(1);
  return(0);
}

////////////////////////////////////////////////////////////////
int G__blockscope::access(/* const */ G__DataMemberInfo& x) const {
  if(access(x.MemberOf()->Tagnum(),x.Property())) return(1);
  return(0);
}

////////////////////////////////////////////////////////////////
int G__blockscope::access(int tagnum,long property) const {

  // member is public
  if(property&G__BIT_ISPUBLIC) return(1);

  // member is protected and this is public inheritance
  if((property&G__BIT_ISPROTECTED) && -1!=tagnum && -1!=m_ifunc->tagnum
     && -1!=G__ispublicbase(tagnum,m_ifunc->tagnum,G__STATICRESOLUTION)) {
    return(1);
  }

  // own class member or // friend class or function
  if(isfriend(tagnum)) return(1);
  return(0);
}

////////////////////////////////////////////////////////////////
int G__blockscope::isfriend(int tagnum) const {
  if (!m_ifunc) return 0;

  int m_tagnum=m_ifunc->tagnum;

  // exactly same class
  if(m_tagnum==tagnum) return(1);

  struct G__friendtag *friendtag;
  if(0<=m_tagnum) { // in member function 
    friendtag = G__struct.friendtag[m_tagnum];
    while(friendtag) {
      if(friendtag->tagnum==tagnum) return(1);
      friendtag=friendtag->next;
    }
  }
  if(-1!=m_iexist) {
    G__ifunc_table_internal* ifunc = G__get_ifunc_internal(m_ifunc);
    friendtag = ifunc->friendtag[m_iexist];
    while(friendtag) {
      if(friendtag->tagnum==tagnum) return(1);
      friendtag=friendtag->next;
    }
  }
  return(0);
}
////////////////////////////////////////////////////////////////
void G__blockscope::Baseclassctor_vbase(int tagnum) {
  // call this function after SETSTROS/NEWALLOC and before LD_FUNC ctor call
  // generate instruction for setting virtual base offset
  //  xxVVVV        yyvvvv
  //  AAAAAAAA ???? BBBBBBBB
  //  DDDDDDDDDDDDDDDDDDDDDDDDDD
  //  |------------>| baseoffset of B. (static)
  //    |<----------| virtual base offset of B. Contents of yy (dynamic)
  int store_pc= m_bc_inst.GetPC();
  G__ClassInfo cls(tagnum);
  G__BaseClassInfo bas(cls);
  map<long,long> vbasetable;
  map<long,long> adrtable;
  while(bas.Next(0)) { // iterate all inheritance
    if(bas.Property()&G__BIT_ISVIRTUALBASE) {
      store_pc = -1; // reset this as a flag 
      if(0==adrtable[bas.Tagnum()]) {
	// the first appearance of virtual base object. 
	vbasetable[bas.Offset()] = G__DOUBLEALLOC;
	adrtable[bas.Tagnum()] = bas.Offset() + G__DOUBLEALLOC;
      }
      else {
	// ghost area of virtual base object
	vbasetable[bas.Offset()] = adrtable[bas.Tagnum()] - bas.Offset();
      }
    }
  }

  if(-1 == store_pc) m_bc_inst.MEMSETINT(1,vbasetable);
}

/****************************************************************
* G__value G__bc_new_operator()
****************************************************************/
extern "C" void G__bc_Baseclassctor_vbase(int tagnum) {
  G__currentscope->Baseclassctor_vbase(tagnum);
}

////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
extern "C" int G__Isconversionctor(G__TypeReader& ltype
				   ,G__TypeReader& rtype) {
  if(ltype.Property()&(G__BIT_ISSTRUCT|G__BIT_ISCLASS)) {
    long dmy;
    string lname = ltype.Name();
    // todo, it is questionable whether I should use 
    // G__ClassInfo::ExactMatch flag to turn off further type conversion.
    G__MethodInfo m = ltype.GetMethod(lname.c_str(),rtype.Name(),&dmy
				      ,G__ClassInfo::ExactMatch
				      //,G__ClassInfo::ConversionMatch
				      //,G__ClassInfo::ConversionMatchBytecode
				      );
    if(m.IsValid()) return(1);
    return(0);
  }
  else return(0);
}

////////////////////////////////////////////////////////////////
extern "C" int G__Isassignmentopr(G__TypeReader& ltype
				  ,G__TypeReader& rtype) {
  if(ltype.Property()&(G__BIT_ISSTRUCT|G__BIT_ISCLASS)) {
    long dmy;
    // todo, it is questionable whether I should use 
    // G__ClassInfo::ExactMatch flag to turn off further type conversion.
    G__MethodInfo m = ltype.GetMethod("operator=",rtype.Name(),&dmy
				      ,G__ClassInfo::ExactMatch
				      //,G__ClassInfo::ConversionMatch
				      //,G__ClassInfo::ConversionMatchBytecode
				      );
    if(m.IsValid()) return(1);
    return(0);
  }
  else return(0);
}

////////////////////////////////////////////////////////////////
extern "C" int G__Isconversionopr(G__TypeReader& ltype
				  ,G__TypeReader& rtype) {
  if(rtype.Property()&(G__BIT_ISSTRUCT|G__BIT_ISCLASS)) {
    string fname = "operator ";
    fname.append(ltype.Name());
    long dmy;
    // todo, it is questionable whether I should use 
    // G__ClassInfo::ExactMatch flag to turn off further type conversion.
    G__MethodInfo m = rtype.GetMethod(fname.c_str(),"",&dmy
				      ,G__ClassInfo::ExactMatch
				      //,G__ClassInfo::ConversionMatch
				      //,G__ClassInfo::ConversionMatchBytecode
				      );
    if(m.IsValid()) return(1);
    return(0);
  }
  else return(0);
}

////////////////////////////////////////////////////////////////
int G__bc_baseconversion(G__value& /*result*/,struct G__var_array* /*var*/
			 ,int /*ig15*/,int /*vartype*/,int /*paran*/) {
  return(0);
}
////////////////////////////////////////////////////////////////
extern "C" int G__Isvalidassignment(G__TypeReader& ltype
				    ,G__TypeReader& rtype,G__value *rval) {
  if(ltype.Ispointer()) {
    if(!rtype.Ispointer()) {
      if(0==G__int(*rval)
	 //&&0==(rval->isconst&G__STATICCONST)
	 ) return(1);//fine
      return(G__Isconversionopr(ltype,rtype));
    }
    if(ltype.Type()==rtype.Type() && ltype.Ispointer()==rtype.Ispointer()) {
      if('U'==ltype.Type()) {
        if(ltype.Tagnum()==rtype.Tagnum()) return(1); // fine
        if(-1!=G__ispublicbase(ltype.Tagnum(),rtype.Tagnum(),(long)0)) {
          return(1); // fine
	}
        else 
	  return(0); // error
      }
      else return(1); // fine
    }
    else if('Y'==ltype.Type()) return(1); // fine
    else return(0); // error
  }
  else {
    if(ltype.Type()==rtype.Type() && ltype.Ispointer()==rtype.Ispointer()) {
      if('u'==ltype.Type()) {
	if(ltype.Tagnum()==rtype.Tagnum()) return(1); 
	// This case may not be needed because ltype has to be fundamental type
	// only except that this is called from var.c
	if(G__Isconversionctor(ltype,rtype)) return(1);
	if(G__Isassignmentopr(ltype,rtype)) return(1);
	return(G__Isconversionopr(ltype,rtype));
      }
      else return(1); // fine
    }
    else if(!ltype.Ispointer() && !rtype.Ispointer()) {
      switch(ltype.Type()) {
      case 'g':
      case 'c': case 's': case 'i': case 'l':
      case 'b': case 'r': case 'h': case 'k':
      case 'f': case 'd':
      case 'n': case 'm':
      case 'q':
        switch(ltype.Type()) {
        case 'g':
        case 'c': case 's': case 'i': case 'l':
        case 'b': case 'r': case 'h': case 'k':
        case 'f': case 'd':
	case 'n': case 'm':
	case 'q':
          return(1); // fine
        default:
	  return(G__Isconversionopr(ltype,rtype));
        }
        break;
      default:
	return(G__Isconversionopr(ltype,rtype));
        break;
      }
    }
    else {
      return(G__Isconversionopr(ltype,rtype));
    }
  }

  return(1);
}

////////////////////////////////////////////////////////////////
extern "C" int G__Isvalidassignment_val(G__value *ltype,int varparan
					,int lparan,int lvar_type
					,G__value *rtype) {
  G__TypeReader ltype1(*ltype);
  int paran = varparan - lparan;
  if(paran>0) for(int i=0;i<paran;i++) ltype1.incplevel();
  else for(int i=0;i<(-paran);i++) ltype1.decplevel();
  if(ltype1.Ispointer() && 'v'==lvar_type) ltype1.decplevel();
  G__TypeReader rtype1(*rtype);
  return(G__Isvalidassignment(ltype1,rtype1,rtype));
}

////////////////////////////////////////////////////////////////
extern "C" int G__bc_conversion(G__value *result
				,struct G__var_array* var,int ig15
				,int var_type,int paran) {
  return(G__currentscope->conversion(*result,var,ig15,var_type,paran));
}


////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
//  Generate instructions
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////
extern "C" void G__bc_VIRTUALADDSTROS(int tagnum
				      ,struct G__inheritance* baseclass
				      ,int basen) {
  G__currentscope->GetInst().VIRTUALADDSTROS(tagnum,baseclass,basen);
}

////////////////////////////////////////////////////////////////
extern "C" void G__bc_cancel_VIRTUALADDSTROS() {
  G__currentscope->GetInst().cancel_VIRTUALADDSTROS();
}

////////////////////////////////////////////////////////////////
extern "C" void G__bc_REWINDSTACK(int n) {
  G__currentscope->GetInst().REWINDSTACK(n);
}

////////////////////////////////////////////////////////////////
int G__casetable::jump(int val) {
  map<long,long>::iterator pos = m_casetable.find((long)val);
  if(pos==m_casetable.end()) return(m_default);
  else return((*pos).second);
}

////////////////////////////////////////////////////////////////
extern "C" int G__bc_casejump(void* p,int val) {
  G__casetable *pcasetable =(G__casetable*)p;
  return(pcasetable->jump(val));
}

////////////////////////////////////////////////////////////////
