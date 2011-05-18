# Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
#
# Permission to use, copy, modify, and distribute this software for any
# purpose is hereby granted without fee, provided that this copyright and
# permissions notice appear in all copies and derivatives.
#
# This software is provided "as is" without express or implied warranty.

""" GCC C++ name demangle python module

   Reference: http://www.codesourcery.com/cxx-abi/abi.html
   
   The encoding is formalized as a derivation grammar along with the explanatory text, 
   in a modified BNF with the following conventions: 

   - Non-terminals are delimited by diamond braces: "<>". 
   - spaces are to be ignored. 
   - Text beginning with '#' is comments, to be ignored. 
   - Tokens in square brackets "[]" are optional. 
   - Tokens are placed in parentheses "()" for grouping purposes. 
   - '*' repeats the preceding item 0 or more times. 
   - '+' repeats the preceding item 1 or more times. 
   - All other characters are terminals, representing themselves. 

    <mangled-name> ::= _Z <encoding>
        <encoding> ::= <function name> <bare-function-type>
                   ::= <data name>
                   ::= <special-name>
              
            <name> ::= <nested-name>
                   ::= <unscoped-name>
                   ::= <unscoped-template-name> <template-args>
                   ::= <local-name>    # See Scope Encoding below

   <unscoped-name> ::= <unqualified-name>
                   ::= St <unqualified-name>   # ::std::

  <unscoped-template-name> ::= <unscoped-name>
                           ::= <substitution>

     <nested-name> ::= N [<CV-qualifiers>] <prefix> <unqualified-name> E
                   ::= N [<CV-qualifiers>] <template-prefix> <template-args> E

          <prefix> ::= <prefix> <unqualified-name>
                   ::= <template-prefix> <template-args>
                   ::= <template-param>
                   ::= # empty
                   ::= <substitution>

 <template-prefix> ::= <prefix> <template unqualified-name>
                   ::= <template-param>
                   ::= <substitution>

<unqualified-name> ::= <operator-name>
                   ::= <ctor-dtor-name>  
                   ::= <source-name>   

     <source-name> ::= <positive length number> <identifier>
          <number> ::= [n] <non-negative decimal integer>
      <identifier> ::= <unqualified source code identifier>

   <operator-name> ::= nw          # new           
                   ::= na          # new[]
                   ::= dl          # delete        
                         ...
                         
    <special-name> ::= TV <type>          # virtual table
                   ::= TT <type>          # VTT structure (construction vtable index)
                   ::= TI <type>          # typeinfo structure
                   ::= TS <type>          # typeinfo name (null-terminated byte string)

    <special-name> ::= T <call-offset> <base encoding>
                         # base is the nominal target function of thunk
     <call-offset> ::= h <nv-offset> _
                   ::= v <v-offset> _
       <nv-offset> ::= <offset number>    # non-virtual base override
        <v-offset> ::= <offset number> _ <virtual offset number>
                         # virtual base override, with vcall offset

    <special-name> ::= Tc <call-offset> <call-offset> <base encoding>
                            # base is the nominal target function of thunk
                            # first call-offset is 'this' adjustment
                            # second call-offset is result adjustment

  <ctor-dtor-name> ::= C1          # complete object constructor
                   ::= C2          # base object constructor
                   ::= C3          # complete object allocating constructor
                   ::= D0          # deleting destructor
                   ::= D1          # complete object destructor
                   ::= D2          # base object destructor

            <type> ::= <builtin-type>
                   ::= <function-type>
                   ::= <class-enum-type>
                   ::= <array-type>
                   ::= <pointer-to-member-type>
                   ::= <template-param>
                   ::= <template-template-param> <template-args>
                   ::= <substitution> # See Compression below

            <type> ::= <CV-qualifiers> <type>
                   ::= P <type>          # pointer-to
                   ::= R <type>          # reference-to
                   ::= C <type>          # complex pair (C 2000)
                   ::= G <type>          # imaginary (C 2000)
                   ::= U <source-name> <type>          # vendor extended type qualifier

   <CV-qualifiers> ::= [r] [V] [K]           # restrict (C99), volatile, const   

    <builtin-type> ::= v          # void
                   ::= w          # wchar_t
                   ::= b          # bool
                   ::= c          # char
                   ::= a          # signed char
                   ::= h          # unsigned char
                   ::= s          # short
                   ::= t          # unsigned short
                   ::= i          # int
                   ::= j          # unsigned int
                   ::= l          # long
                   ::= m          # unsigned long
                   ::= x          # long long, __int64
                   ::= y          # unsigned long long, __int64
                   ::= n          # __int128
                   ::= o          # unsigned __int128
                   ::= f          # float
                   ::= d          # double
                   ::= e          # long double, __float80
                   ::= g          # __float128
                   ::= z          # ellipsis
                   ::= u <source-name>          # vendor extended type

   <function-type> ::= F [Y] <bare-function-type> E
<bare-function-type> ::= <signature type>+
                           # types are possible return type, then parameter types

 <class-enum-type> ::= <name>
      <array-type> ::= A <positive dimension number> _ <element type>
                   ::= A [<dimension expression>] _ <element type>

 <pointer-to-member-type> ::= M <class type> <member type>

  <template-param> ::= T_          # first template parameter
                   ::= T <parameter-2 non-negative number> _
  <template-template-param> ::= <template-param>
                            ::= <substitution>

   <template-args> ::= I <template-arg>+ E
    <template-arg> ::= <type>                     # type or template
                   ::= X <expression> E           # expression
                   ::= <expr-primary>             # simple expressions

      <expression> ::= <unary operator-name> <expression>
                   ::= <binary operator-name> <expression> <expression>
                   ::= <trinary operator-name> <expression> <expression> <expression>
                   ::= st <type>
                   ::= <template-param>
                   ::= sr <type> <unqualified-name>                   # dependent name
                   ::= sr <type> <unqualified-name> <template-args>   # dependent template-id
                   ::= <expr-primary>

    <expr-primary> ::= L <type> <value number> E                   # integer literal
                   ::= L <type <value float> E                     # floating literal
                   ::= L <mangled-name> E                          # external name

     <local-name>  ::= Z <function encoding> E <entity name> [<discriminator>]
                   ::= Z <function encoding> E s [<discriminator>]
   <discriminator> ::= _ <non-negative number> 

   <substitution>  ::= S <seq-id> _
                   ::= S_

    <substitution> ::= St # ::std::
    <substitution> ::= Sa # ::std::allocator
    <substitution> ::= Sb # ::std::basic_string
    <substitution> ::= Ss # ::std::basic_string < char,::std::char_traits<char>,::std::allocator<char> >
    <substitution> ::= Si # ::std::basic_istream<char,  std::char_traits<char> >
    <substitution> ::= So # ::std::basic_ostream<char,  std::char_traits<char> >
    <substitution> ::= Sd # ::std::basic_iostream<char, std::char_traits<char> >

            <name> ::= St <unqualified-name> # ::std::


"""
debug = False
from string import join

basictypes = { 'v':'void', 'w':'wchar_t', 'b':'bool', 'c':'char', 
               'h':'unsigned char', 'a':'signed char',
               'i':'int', 'j':'unsigned', 's':'short', 't':'unsigned short',
               'l':'long', 'm':'unsigned long', 
               'x':'long long','y':'unsigned long long',
               'f':'float', 'd':'double', 'e':'long double' }
basicquali = { 'K':'const', 'V':'volatile'}
operators  = { 'aN':('&=', 2 ), 'aS': ('=', 2), 'aa': ('&&', 2), 'ad': ('&', 1), 'an': ('&', 2),
               'cl': ('()' , 0), 'cm': (',', 2), 'co': ('~', 1), 'dV': ('/=' , 2), 
               'da': (' delete[]', 1), 'de': ('*', 1), 'dl': (' delete', 1), 'dv': ('/', 2),
               'eO': ('^=' , 2), 'eo': ('^', 2), 'eq': ('==' , 2), 
               'ge': ('>=' , 2), 'gt': ('>', 2), 'ix': ('[]' , 2), 'lS': ('<<=', 2), 'le': ('<=' , 2),
               'ls': ('<<' , 2), 'lt': ('<', 2), 'mI': ('-=' , 2), 'mL': ('*=' , 2), 'mi': ('-', 2), 
               'ml': ('*', 2), 'mm': ('--' , 1), 'na': (' new[]' , 1), 'ne': ('!=' , 2), 'ng': ('-', 1), 
               'nt': ('!', 1), 'nw': (' new' , 1), 'oR': ('|=' , 2), 'oo': ('||' , 2), 'or': ('|', 2), 
               'pL': ('+=' , 2), 'pl': ('+', 2), 'pm': ('->*', 2), 'pp': ('++' , 1), 'ps': ('+', 1), 
               'qu': ('?', 3), 'rM': ('%=' , 2), 'rS': ('>>=', 2), 'rm': ('%', 2), 
               'rs': ('>>' , 2), 'sz': (' sizeof', 1) } 
symbols       = []
last_name     = ''

def demangle_mangled_name(name) :
  if debug : print 'demangle_mangled_name ', name[:50] 
  if name[0:2] == '_Z' :
    r = demangle_encoding(name[2:])
    return r[0]+2, r[1]
  else :
    raise 'Demangling error'

def demangle_encoding(name) :
  if debug : print 'demangle_encoding ', name[:50]
  # return (size, name, modifiers, istemplated)
  if name[0] in ('T', 'G' ) :
    return demangle_special_name(name)
  else : 
    re = _demangle_name(name)
    if len(name) > re[0] and name[re[0]] != 'E' :
      # returns (size, ret, args)
      rf = demangle_function(name[re[0]:],re[3])
      return re[0]+rf[0], rf[1]+re[1]+rf[2]+re[2]    
    else :
      return re[0], re[1]

def _demangle_name(name) :
  if debug : print '_demangle_name ', name[:50]
  c = name[0]
  istemplate = False
  modifiers  = ''
  if c == 'N' :
    # returns (size, name, modifiers, istemplated)
    r = demangle_nested_name(name)
    modifiers  = r[2]
    istemplate = r[3]
  elif c == 'Z' :
    r = demangle_local_name(name)
  else :
    r = demangle_unscoped_name(name)
    if r[0] < len(name) and name[r[0]] == 'I':
      istemplate = True
      re = demangle_template_args(name[r[0]:])
      r = r[0]+re[0], r[1]+re[1]
  return r[0], r[1], modifiers, istemplate

def demangle_local_name(name) :
  if debug : print 'demangle_local_name ', name[:50]
  re = demangle_encoding(name[1:])
  r = 1 + re[0] , re[1]
  if name[r[0]] == 'E' :
    re = _demangle_name(name[r[0]+1:])
    r = r[0]+ 1 + re[0], r[1]+re[1]
  return r

def demangle_unscoped_name(name) :
  if debug : print 'demangle_unscope_name ', name[:50]
  if name[0:2] == 'St' :
    re = demangle_unqualified_name(name[2:])
    r = 2 + re[0], 'std::' + re[1]
  elif name[0] == 'S' :
    r = demangle_substitution(name) 
  else :
    r = demangle_unqualified_name(name)
  add_symbol(r[1])
  return r
    
def demangle_nested_name(name) :
  if debug : print 'demangle_nested_name ', name[:50]
  i = 1
  s = ''
  if name[i] in basicquali :
    s = ' '+ basicquali[name[i]]
    i += 1
  e = []
  #t = ''
  while name[i] != 'E' :
    istemplate = False
    isctordtor = False
    if name[i] == 'S' :
      re = demangle_substitution(name[i:])
      if re[0]+i < len(name) and name[re[0]+i] == 'I' :
        add_symbol(join(e+[re[1]],'::'))
        istemplate = True
        rt = demangle_template_args(name[re[0]+i:])
        re = re[0]+rt[0], re[1]+rt[1] 
      e += [re[1]]
    else :
      re = demangle_unqualified_name(name[i:])
      isctordtor = re[2]
      lname = re[1] 
      if re[0]+i < len(name) and name[re[0]+i] == 'I' :
        add_symbol(join(e+[re[1]],'::'))
        istemplate = True
        rt = demangle_template_args(name[re[0]+i:])
        re = re[0]+rt[0], re[1]+rt[1] 
      e += [re[1]]
      global last_name
      last_name = lname; 'last_name = ', last_name
    i += re[0]
    if re[1][0:8] != 'operator' : add_symbol(join(e,'::'))
  return  i + 1, join(e,'::'), s, istemplate and not isctordtor
  
def demangle_unqualified_name(name):
  if debug : print 'demangle_unqualified_name ', name[:50]
  c = name[0]
  isctordtor = False
  if c.isdigit() : 
    i = 0; n = 0
    while name[i].isdigit() : 
      n *= 10; n += int(name[i]); i += 1
    r = i + n, name[i:i+n]
    if r[1][0:11] == '_GLOBAL__N_' : 
      r = i + n, '(anonymous namespace)'
  elif c >= 'a' and c <= 'z' : 
    r = demangle_operator(name, False)
  elif c in ('C', 'D') :
    isctordtor = True
    r = demangle_ctor_dtor(name)
  return r[0], r[1], isctordtor

def demangle_operator(name, shortname ) :
  if debug : print 'demangle_operator ', name[:50]
  if name[0:2] == 'cv' :
    re = demangle_type(name[2:])
    return re[0]+2 , 'operator '+re[1], 0
  if name[0:2] in operators :
    if shortname :
      return 2, operators[name[0:2]][0], operators[name[0:2]][1]
    else :
      return 2, 'operator'+operators[name[0:2]][0], operators[name[0:2]][1]
  else :
    print 'unknown operator'

ctor_flavors = ('in-charge', 'not-in-charge', 'in-charge allocating', 'not-in-charge allocating')
dtor_flavors = ('in-charge deleting','in-charge','not-in-charge')
def demangle_ctor_dtor(name) :
  if debug : print 'demangle_ctor_dtor ', name[:50]
  if name[0] == 'C' :
    return 2,last_name
  elif name[0] == 'D' :
    return 2, '~'+last_name

specialnames1 = { 'TV':'virtual table', 'TT':'VTT structure',
                  'TI':'typeinfo', 'TS':'typeinfo name'}
def demangle_special_name(name) :
  if debug : print 'demangle_special_name ', name[:50]
  if name[0:2] in specialnames1 :
    r = demangle_type(name[2:])
    return r[0]+2, specialnames1[name[0:2]] +' '+ r[1]
  elif name[0:2] == 'GV' :
    r = _demangle_name(name[2:])
    return r[0]+2, 'guard variable for ' + r[1]
  elif name[0] == 'T' :
    raise 'Not implemented'
      
def demangle_template_args(name) :
  if debug : print 'demangle_template_args ', name[:50]
  i = 1
  e = []
  while name[i] != 'E' :
    re = demangle_template_arg(name[i:])
    e += [re[1]]
    i += re[0]
    if e[-1][-1] == '>' : e[-1] += ' '
    r = i + 1, '<' + join(e, ', ') + '>'
  add_template_arg_list(e)
  return r

def demangle_template_arg(name) :
  if debug : print 'demangle_template_arg ', name[:50]
  
  if name[0] == 'L':
    r = demangle_literal(name)
  elif name[0] == 'X':
    re = demangle_expression(name[1:])
    r = re[0]+1, re[1]
  else :
    r = demangle_type(name)
  return r

def demangle_template_param(name) :
  if debug : print 'demangle_template_param ', name[:50]
  if name[0] == 'T' :
    if name[1] == '_' :
      return 2, get_template_arg(0)
    else :
      i = 1; n = 0
      while name[i].isdigit() :
        n *= 10; n += int(name[i]); i += 1
      return i+1, get_template_arg(n+1)

def demangle_literal(name):
  if debug : print 'demangle_literal ', name[:50]
  if name[1] >= 'a' and name[1] <= 'z' :
    if name[1] == 'b' :
      if name[2] == '0'   : return 4, 'false'
      elif name[2] == '1' : return 4, 'true'
    if name[1] in ('i', 'l') :
      i = 2
      n = 0
      while name[i].isdigit() :
        n *= 10; n += int(name[i]); i += 1
      return  i+1, '%d' % n
  else :
    i = 1
    n = 0
    while name[i].isdigit() :
      n *= 10; n += int(name[i]); i += 1
    return  i+1, '%d' % n

def demangle_expression(name):
  if debug : print 'demangle_expression ', name[:50]
  if name[0:2] == 'sr' :
    rt = demangle_type(name[2:])
    rn = demangle_encoding(name[2+rt[0]:])
    return 2+rt[0]+rn[0],rt[1]+'::'+rn[1]
  elif name[0] == 'T' :
    rt = demangle_template_param(name[1:])
    return 1+rt[0], rt[1]
  elif name[0] == 'L' :
    if name[1] == '_' :
      rt = demangle_mangled_name(name[1:])
    else :
      rt = demangle_literal(name[1:])
    return 2+rt[0],rt[1]
  else :
    ro = demangle_operator(name, True)
    if ro[2] == 1 :
      ra1 = demangle_expression(name[ro[0]:])
      return ro[0]+ra1[0] , ro[1]+'('+ra1[1]+')'
    elif ro[2] == 2 :
      ra1 = demangle_expression(name[ro[0]:])
      ra2 = demangle_expression(name[ro[0]+ra1[0]:])
      return ro[0]+ra1[0]+ra2[0] , ro[1]+'('+ra1[1]+')'+'('+ra2[1]+')'
    elif ro[2] == 3 :
      ra1 = demangle_expression(name[ro[0]:])
      ra2 = demangle_expression(name[ro[0]+ra1[0]:])
      ra3 = demangle_expression(name[ro[0]+ra1[0]+ra2[0]:])
      return ro[0]+ra1[0]+ra2[0]+ra3[0] , ro[1]+'('+ra1[1]+')'+'('+ra2[1]+')'+':('+ra3[1]+')'

def demangle_type(name) :
  if debug : print 'demangle_type ', name[:50]
  c = name[0]
  if c.isdigit() or c == 'N' or c == 'Z' :
    r = _demangle_name(name)
  elif c in basictypes :
    r =  1, basictypes[c]
  else :  
    if c in basicquali :
      i = 0; q = []
      while( name[i] in basicquali ) :
        q += [basicquali[name[i]]]
        i += 1
      rt = demangle_type(name[i:])
      if name[i] not in ('P', 'R') :
         r = rt[0] + i, rt[1] +' '+ join(q, ' ')
      else :
         r = rt[0] + i, join(q,' ') +' '+ rt[1]
    elif c in ('P','R','M') :
      r = demangle_type_ptr(name)
    elif c == 'A' :
      i = 1; n = 0
      while name[i].isdigit() :
        n *= 10; n += int(name[i]); i += 1
      rt = demangle_type(name[i+1:])
      r = i + 1 + rt[0], '%s[%d]'%(rt[1],n)
    elif c == 'T':
      r = demangle_template_param (name)
    elif c == 'S':
      if name[1].isdigit() or name[1] == '_' :
        r = demangle_substitution(name)
      else :
        r = _demangle_name(name)
    else :
      print 'Not found type ', c
  if len(name) > r[0] and name[r[0]] == 'I' :
    re = demangle_template_args(name[r[0]:])
    r = r[0]+re[0], r[1]+re[1]
  if c not in basictypes :  add_symbol(r[1])
  return r
  
def demangle_type_ptr(name) :
  if debug : print 'demangle_type_ptr ', name[:50]
  s = ''
  i = 0
  while(i < len(name)) :
    c = name[i]
    if   c == 'P' : s += '*'
    elif c == 'R' : s += '&'
    elif c == 'M' :
      re = demangle_type(name[i+1:])
      s = re[1]+'::*'+s
      i += re[0]+1
      if name[i] == 'F' : continue
      else :
        rt = demangle_type(name[i:])
        return rt[0]+i, rt[1]+' '+s
    elif c == 'F' :
      re = demangle_function(name[i+1:])
      return re[0]+i+1, re[1]+re[2][:re[2].find('(')] + '(' + s + ')' + re[2][re[2].find('('):]
    else :
      rt = demangle_type(name[i:])
      return rt[0] + i, rt[1] + s
    i += 1
    
def demangle_function(name, rtype=True) :
  if debug : print 'demangle_function ', name[:50]
  i = 0
  e = []
  while len(name) > i and name[i] != 'E' :
    re = demangle_type(name[i:])
    e += [re[1]]
    i += re[0]
  if rtype :
    if len(e) == 2 and e[1] == 'void' :
      r = i + 1, e[0]+' ', '()'
    else :
      r = i + 1, e[0]+' ', '('+join(e[1:],', ')+')'
  else :
    if len(e) == 1 and e[0] == 'void' :
      r = i + 1, '', '()'
    else :
      r = i + 1, '', '('+join(e[0:],', ')+')'
  return r

def demangle_member(name) :
  if debug : print 'demangle_member ', name[:50]
  if name[0] == 'F' :
    i = 1
    e = []
    while name[i] != 'E' :
      ii, re = demangle_type(name[i:])
      e += [re]
      i += ii
    if len(e) == 2 and e[1] == 'void' :
      r = i + 1, e[0]+' ()()'
    else :
      r = i + 1, e[0]+' ()('+join(e[1:], ', ')+')'
    return r

def demangle_substitution(name) :
  if debug : print 'demangle_substitution ', name[:50]
  if name[1] == 't' :
    re = _demangle_name(name[2:])
    r = 2 + re[0], 'std::' + re[1]
  elif name[1] == 'a' : r = 2, 'std::allocator'
  elif name[1] == 'b' : r = 2, 'std::basic_string'
  elif name[1] == 's' : r = 2, 'std::basic_string<char, std::char_traits<char>, std::allocator<char> >'
  elif name[1] == 'i' : r = 2, 'std::basic_istream<char, std::char_traits<char> >'
  elif name[1] == 'o' : r = 2, 'std::basic_ostream<char, std::char_traits<char> >'
  elif name[1] == 'd' : r = 2, 'std::basic_iostream<char, std::char_traits<char> >'
  else :
    i = 1; n = 0; s = 0
    while name[i].isdigit() or (name[i] >= 'A' and name[i] <= 'Z') :
      n *= 36
      if name[i].isdigit() : n += int(name[i])
      else                 : n += 10 + ord(name[i])-ord('A')
      i += 1
      s = 1
    r = i + 1, get_symbol(s+n)
    add_symbol(r[1])
  return r
 
def add_symbol(s) :
  global symbols
  if s not in symbols :
    if debug : print 'adding symbol[%d] = %s' % (len(symbols), s)
    symbols.append(s)
def get_symbol(i) :
  global symbols
  if i < len(symbols) :
    return symbols[i]
  else :
    print 'symbol %d not found' % i
    return ''
def add_template_arg_list(l) :
  global temparglist
  temparglist = l
  if debug : print 'targlist = ', temparglist
def get_template_arg(i) :
  global temparglist
  return temparglist[i]
def clear_symbols() :
  global symbols
  symbols = []

def demangle( name ):
  clear_symbols()
  if name[0:2] == '_Z' :
    r = demangle_encoding(name[2:])
  else :
    r = demangle_type(name)
  return r[1] 

def demangle_name( name ):
  clear_symbols()
  return _demangle_name( name )

if __name__ == '__main__' :
  cases = [
('c', 'char'),
('Pc', 'char*'),
('PKc', 'char const*'),
('PVKc', 'char volatile const*'),
('A20_c', 'char[20]'),
('h', 'unsigned char'),
('a', 'signed char'),
('i', 'int'),
('j', 'unsigned'),
('Pj', 'unsigned*'),
('PKj', 'unsigned const*'),
('PVj', 'unsigned volatile*'),
('PVKj', 'unsigned volatile const*'),
('s', 'short'),
('l', 'long'),
('x', 'long long'),
('f', 'float'),
('d', 'double'),
('e', 'long double'),
('6foobar', 'foobar'),
('N6foobar3bazE', 'foobar::baz'),
('N3foo3barE', 'foo::bar'),
('N3foo3bazE', 'foo::baz'),
('N3foo5young3eggE', 'foo::young::egg'),
('PF1xS_E', 'x (*)(x)'),
('PF1x2xxS0_E', 'x (*)(xx, xx)'),
('PF1x2xx3xxxS1_E', 'x (*)(xx, xxx, xxx)'),
('PF1x2xx3xxx4xxxxS2_E', 'x (*)(xx, xxx, xxxx, xxxx)'),
('PFvvE', 'void (*)()'),
('PFivE', 'int (*)()'),
('PFRivE', 'int& (*)()'),
('PFviE', 'void (*)(int)'),
('PFvRiE', 'void (*)(int&)'),
('PFvRKiE', 'void (*)(int const&)'),
('PFvRVKiE', 'void (*)(int volatile const&)'),
('PFvi6foobarE', 'void (*)(int, foobar)'),
('M6foobarFN3foo3bazEiS_E', 'foo::baz (foobar::*)(int, foobar)'),
('8TemplateIlPcLl42ELi99EE', 'Template<long, char*, 42, 99>'),
('_ZN7Complex16TemplatedMethodsC1IdEERKT_','Complex::TemplatedMethods::TemplatedMethods<double>(double const&)'),
('St4pairIiS_IlcEE', 'std::pair<int, std::pair<long, char> >'),
('St4pairI6foobarS_IS0_S_IN3foo3bazES2_EEE', 'std::pair<foobar, std::pair<foobar, std::pair<foo::baz, foo::baz> > >'),
('11TemplateTwoIXadL_Z1fvEEE', 'TemplateTwo<&(f())>'),
('13TemplateThreeIXadsr6foobarNS0_1fEvEE', 'TemplateThree<&(foobar::foobar::f())>'),
('_ZNSt15basic_streambufIcSt11char_traitsIcEE7_MylockE', 'std::basic_streambuf<char, std::char_traits<char> >::_Mylock'),
('_ZNK8TMethods3getIiEET_v', 'int TMethods::get<int>() const'),
('St11_Vector_valIPiSaIS0_EE', 'std::_Vector_val<int*, std::allocator<int*> >'),
('St6vectorIPKiSaIS1_EE', 'std::vector<int const*, std::allocator<int const*> >'),
('_ZN6foobar4funcEil', 'foobar::func(int, long)'),
('_ZN7Complex16TemplatedMethods3setIfEEvRKT_','void Complex::TemplatedMethods::set<float>(float const&)'),
('_Z41__static_initialization_and_destruction_0ii', '__static_initialization_and_destruction_0(int, int)'),
('_ZN9__gnu_cxxneIPN5boost18default_color_typeESt6vectorIS2_SaIS2_EEEEbRKNS_17__normal_iteratorIT_T0_EESC_', 'bool __gnu_cxx::operator!=<boost::default_color_type*, std::vector<boost::default_color_type, std::allocator<boost::default_color_type> > >(__gnu_cxx::__normal_iterator<boost::default_color_type*, std::vector<boost::default_color_type, std::allocator<boost::default_color_type> > > const&, __gnu_cxx::__normal_iterator<boost::default_color_type*, std::vector<boost::default_color_type, std::allocator<boost::default_color_type> > > const&)'),
#('_ZNK5boost6python3api16object_operatorsINS1_5proxyINS1_18attribute_policiesEEEEclINS1_6objectES8_S8_bbS8_EENS0_6detail9dependentIS8_T_E4typeERKSB_RKT0_RKT1_RKT2_RKT3_RKT4_', 'boost::python::detail::dependent<boost::python::api::object, boost::python::api::object>::type boost::python::api::object_operators<boost::python::api::proxy<boost::python::api::attribute_policies> >::operator()<boost::python::api::object, boost::python::api::object, boost::python::api::object, bool, bool, boost::python::api::object>(boost::python::api::object const&, boost::python::api::object const&, boost::python::api::object const&, bool const&, bool const&, boost::python::api::object const&) const'),('_ZN5boostneIjjNS_26counting_iterator_policiesIjEEjjRKjS4_PS3_S5_St26random_access_iterator_tagxEEbRKNS_16iterator_adaptorIT_T1_T2_T4_T6_T8_T9_EERKNS7_IT0_S9_T3_T5_T7_SD_SE_EE', 'bool boost::operator!=<unsigned int, unsigned int, boost::counting_iterator_policies<unsigned int>, unsigned int, unsigned int, unsigned int const&, unsigned int const&, unsigned int const*, unsigned int const*, std::random_access_iterator_tag, long long>(boost::iterator_adaptor<unsigned int, boost::counting_iterator_policies<unsigned int>, unsigned int, unsigned int const&, unsigned int const*, std::random_access_iterator_tag, long long> const&, boost::iterator_adaptor<unsigned int, boost::counting_iterator_policies<unsigned int>, unsigned int, unsigned int const&, unsigned int const*, std::random_access_iterator_tag, long long> const&)'),

#('_ZN5boost9get_paramINS_15graph_visitor_tENS_14buffer_param_tENS_11bfs_visitorINS_17distance_recorderINS_21iterator_property_mapIN9__gnu_cxx17__normal_iteratorIPjSt6vectorIjSaIjEEEENS_26vec_adj_list_vertex_id_mapINS_11no_propertyEjEEjRjEENS_12on_tree_edgeEEEEESE_EENS_14property_valueINS_16bgl_named_paramsIT1_T_T2_EET0_E4typeERKSQ_SR_', 'boost::property_value<boost::bgl_named_params<boost::bfs_visitor<boost::distance_recorder<boost::iterator_property_map<__gnu_cxx::__normal_iterator<unsigned int*, std::vector<unsigned int, std::allocator<unsigned int> > >, boost::vec_adj_list_vertex_id_map<boost::no_property, unsigned int>, unsigned int, unsigned int&>, boost::on_tree_edge> >, boost::graph_visitor_t, boost::no_property>, boost::buffer_param_t>::type boost::get_param<boost::graph_visitor_t, boost::buffer_param_t, boost::bfs_visitor<boost::distance_recorder<boost::iterator_property_map<__gnu_cxx::__normal_iterator<unsigned int*, std::vector<unsigned int, std::allocator<unsigned int> > >, boost::vec_adj_list_vertex_id_map<boost::no_property, unsigned int>, unsigned int, unsigned int&>, boost::on_tree_edge> >, boost::no_property>(boost::bgl_named_params<boost::bfs_visitor<boost::distance_recorder<boost::iterator_property_map<__gnu_cxx::__normal_iterator<unsigned int*, std::vector<unsigned int, std::allocator<unsigned int> > >, boost::vec_adj_list_vertex_id_map<boost::no_property, unsigned int>, unsigned int, unsigned int&>, boost::on_tree_edge> >, boost::graph_visitor_t, boost::no_property> const&, boost::buffer_param_t)'),
('_ZGVN5boost6python9converter6detail15registered_baseIRVKbE10convertersE', 'guard variable for boost::python::converter::detail::registered_base<bool volatile const&>::converters'),
('_ZN52_GLOBAL__N_libs_python_src_object_iterator.cppJrStsb2_1E', '(anonymous namespace)::_1'),
#('_ZN5boost10out_degreeINS_14adjacency_listINS_4vecSES2_NS_14bidirectionalSENS_11no_propertyENS_8propertyINS_12edge_index_tEjNS5_INS_55_GLOBAL__N_libs_python_src_object_inheritance.cpp1VSYrb11edge_cast_tEPFPvS9_ES4_EEEES4_NS_5listSEEERKSF_EEN18BidirectionalGraph16degree_size_typeENSI_17vertex_descriptorERKNS_13reverse_graphISI_T0_EE', 'BidirectionalGraph::degree_size_type boost::out_degree<boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, boost::no_property, boost::property<boost::edge_index_t, unsigned int, boost::property<boost::(anonymous namespace)::edge_cast_t, void* (*)(void*), boost::no_property> >, boost::no_property, boost::listS>, boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, boost::no_property, boost::property<boost::edge_index_t, unsigned int, boost::property<boost::(anonymous namespace)::edge_cast_t, void* (*)(void*), boost::no_property> >, boost::no_property, boost::listS> const&>(BidirectionalGraph::vertex_descriptor, boost::reverse_graph<BidirectionalGraph, boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, boost::no_property, boost::property<boost::edge_index_t, unsigned int, boost::property<boost::(anonymous namespace)::edge_cast_t, void* (*)(void*), boost::no_property> >, boost::no_property, boost::listS> const&> const&)'),
('_ZN5boost10vector_tagC1Ev', 'boost::vector_tag::vector_tag()'),
('_ZN5boost11bfs_visitorINS_12null_visitorEEC1ES1_', 'boost::bfs_visitor<boost::null_visitor>::bfs_visitor(boost::null_visitor)'),
#('_ZN5boost11bfs_visitorINS_17distance_recorderINS_21iterator_property_mapIN9__gnu_cxx17__normal_iteratorIPjSt6vectorIjSaIjEEEENS_26vec_adj_list_vertex_id_mapINS_11no_propertyEjEEjRjEENS_12on_tree_edgeEEEE11gray_targetINS_6detail14edge_desc_implINS_17bidirectional_tagEjEEKNS_13reverse_graphINS_14adjacency_listINS_4vecSESP_NS_14bidirectionalSESB_NS_8propertyINS_12edge_index_tEjNSR_INS_55_GLOBAL__N_libs_python_src_object_inheritance.cpp1VSYrb11edge_cast_tEPFPvSV_ESB_EEEESB_NS_5listSEEERKS11_EEEEvT_RT0_', 'void boost::bfs_visitor<boost::distance_recorder<boost::iterator_property_map<__gnu_cxx::__normal_iterator<unsigned int*, std::vector<unsigned int, std::allocator<unsigned int> > >, boost::vec_adj_list_vertex_id_map<boost::no_property, unsigned int>, unsigned int, unsigned int&>, boost::on_tree_edge> >::gray_target<boost::detail::edge_desc_impl<boost::bidirectional_tag, unsigned int>, boost::reverse_graph<boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, boost::no_property, boost::property<boost::edge_index_t, unsigned int, boost::property<boost::(anonymous namespace)::edge_cast_t, void* (*)(void*), boost::no_property> >, boost::no_property, boost::listS>, boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, boost::no_property, boost::property<boost::edge_index_t, unsigned int, boost::property<boost::(anonymous namespace)::edge_cast_t, void* (*)(void*), boost::no_property> >, boost::no_property, boost::listS> const&> const>(boost::detail::edge_desc_impl<boost::bidirectional_tag, unsigned int>, boost::reverse_graph<boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, boost::no_property, boost::property<boost::edge_index_t, unsigned int, boost::property<boost::(anonymous namespace)::edge_cast_t, void* (*)(void*), boost::no_property> >, boost::no_property, boost::listS>, boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, boost::no_property, boost::property<boost::edge_index_t, unsigned int, boost::property<boost::(anonymous namespace)::edge_cast_t, void* (*)(void*), boost::no_property> >, boost::no_property, boost::listS> const&> const&)'),
('_ZN5boost11no_propertyC1ERKS0_', 'boost::no_property::no_property(boost::no_property const&)'),
('_ZN5boost11noncopyableC2Ev', 'boost::noncopyable::noncopyable()'),
('_ZN5boost11noncopyableD2Ev', 'boost::noncopyable::~noncopyable()'),
#('_ZN5boost12GraphConceptINS_13reverse_graphINS_14adjacency_listINS_4vecSES3_NS_14bidirectionalSENS_11no_propertyENS_8propertyINS_12edge_index_tEjNS6_INS_55_GLOBAL__N_libs_python_src_object_inheritance.cpp1VSYrb11edge_cast_tEPFPvSA_ES5_EEEES5_NS_5listSEEERKSG_EEE11constraintsEv', 'boost::GraphConcept<boost::reverse_graph<boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, boost::no_property, boost::property<boost::edge_index_t, unsigned int, boost::property<boost::(anonymous namespace)::edge_cast_t, void* (*)(void*), boost::no_property> >, boost::no_property, boost::listS>, boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, boost::no_property, boost::property<boost::edge_index_t, unsigned int, boost::property<boost::(anonymous namespace)::edge_cast_t, void* (*)(void*), boost::no_property> >, boost::no_property, boost::listS> const&> >::constraints()'),
#('_ZN5boost12choose_paramINS_11bfs_visitorINS_17distance_recorderINS_21iterator_property_mapIN9__gnu_cxx17__normal_iteratorIPjSt6vectorIjSaIjEEEENS_26vec_adj_list_vertex_id_mapINS_11no_propertyEjEEjRjEENS_12on_tree_edgeEEEEENS1_INS_12null_visitorEEEEERKNS_6detail19choose_param_helperIT_E6result4typeERKSN_RKT0_', 'boost::detail::choose_param_helper<boost::bfs_visitor<boost::distance_recorder<boost::iterator_property_map<__gnu_cxx::__normal_iterator<unsigned int*, std::vector<unsigned int, std::allocator<unsigned int> > >, boost::vec_adj_list_vertex_id_map<boost::no_property, unsigned int>, unsigned int, unsigned int&>, boost::on_tree_edge> > >::result::type const& boost::choose_param<boost::bfs_visitor<boost::distance_recorder<boost::iterator_property_map<__gnu_cxx::__normal_iterator<unsigned int*, std::vector<unsigned int, std::allocator<unsigned int> > >, boost::vec_adj_list_vertex_id_map<boost::no_property, unsigned int>, unsigned int, unsigned int&>, boost::on_tree_edge> >, boost::bfs_visitor<boost::null_visitor> >(boost::bfs_visitor<boost::distance_recorder<boost::iterator_property_map<__gnu_cxx::__normal_iterator<unsigned int*, std::vector<unsigned int, std::allocator<unsigned int> > >, boost::vec_adj_list_vertex_id_map<boost::no_property, unsigned int>, unsigned int, unsigned int&>, boost::on_tree_edge> > const&, boost::bfs_visitor<boost::null_visitor> const&)'),
#('_ZN5boost12choose_paramINS_6detail24error_property_not_foundENS1_8wrap_refINS_5queueIjSt5dequeIjSaIjEEEEEEEERKNS1_19choose_param_helperIT_E6result4typeERKSB_RKT0_', 'boost::detail::choose_param_helper<boost::detail::error_property_not_found>::result::type const& boost::choose_param<boost::detail::error_property_not_found, boost::detail::wrap_ref<boost::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > > >(boost::detail::error_property_not_found const&, boost::detail::wrap_ref<boost::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > > const&)'),
('_ZN5boost12color_traitsINS_18default_color_typeEE4grayEv', 'boost::color_traits<boost::default_color_type>::gray()'),
#('_ZN5boost12num_verticesINS_6detail12adj_list_genINS_14adjacency_listINS_4vecSES4_NS_14bidirectionalSENS_11no_propertyENS_8propertyINS_12edge_index_tEjNS7_INS_55_GLOBAL__N_libs_python_src_object_inheritance.cpp1VSYrb11edge_cast_tEPFPvSB_ES6_EEEES6_NS_5listSEEES4_S4_S5_S6_SF_S6_SG_E6configENS_40bidirectional_graph_helper_with_propertyISJ_EEEEN6Config18vertices_size_typeERKNS_15adj_list_helperISM_T0_EE', 'Config::vertices_size_type boost::num_vertices<boost::detail::adj_list_gen<boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, boost::no_property, boost::property<boost::edge_index_t, unsigned int, boost::property<boost::(anonymous namespace)::edge_cast_t, void* (*)(void*), boost::no_property> >, boost::no_property, boost::listS>, boost::vecS, boost::vecS, boost::bidirectionalS, boost::no_property, boost::property<boost::edge_index_t, unsigned int, boost::property<boost::(anonymous namespace)::edge_cast_t, void* (*)(void*), boost::no_property> >, boost::no_property, boost::listS>::config, boost::bidirectional_graph_helper_with_property<boost::detail::adj_list_gen<boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, boost::no_property, boost::property<boost::edge_index_t, unsigned int, boost::property<boost::(anonymous namespace)::edge_cast_t, void* (*)(void*), boost::no_property> >, boost::no_property, boost::listS>, boost::vecS, boost::vecS, boost::bidirectionalS, boost::no_property, boost::property<boost::edge_index_t, unsigned int, boost::property<boost::(anonymous namespace)::edge_cast_t, void* (*)(void*), boost::no_property> >, boost::no_property, boost::listS>::config> >(boost::adj_list_helper<Config, boost::bidirectional_graph_helper_with_property<boost::detail::adj_list_gen<boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, boost::no_property, boost::property<boost::edge_index_t, unsigned int, boost::property<boost::(anonymous namespace)::edge_cast_t, void* (*)(void*), boost::no_property> >, boost::no_property, boost::listS>, boost::vecS, boost::vecS, boost::bidirectionalS, boost::no_property, boost::property<boost::edge_index_t, unsigned int, boost::property<boost::(anonymous namespace)::edge_cast_t, void* (*)(void*), boost::no_property> >, boost::no_property, boost::listS>::config> > const&)'),
('_ZN5boost12numeric_castIalEET_T0_', 'signed char boost::numeric_cast<signed char, long>(long)'),
]
  t = 0
  l = len(cases)
  for c,r in cases :
    t += 1
    try:
      dname = demangle(c)
      if dname != r :
        print '-%d/%d----ERROR-----with--%s' % (t,l,c)
        print 'Expect> ', r
        print 'Found > ', dname
        debug = True
        demangle(c)
        break
    except:
      print '-%d/%d----FATAL----with--%s' %(t,l,c)
      debug = True
      demangle(c) 
    
