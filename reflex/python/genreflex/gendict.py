# Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
#
# Permission to use, copy, modify, and distribute this software for any
# purpose is hereby granted without fee, provided that this copyright and
# permissions notice appear in all copies and derivatives.
#
# This software is provided "as is" without express or implied warranty.

import xml.parsers.expat
import os, sys, string, time
import gccdemangler
class genDictionary(object) :
#----------------------------------------------------------------------------------
  def __init__(self, hfile, opts):
    self.classes    = []
    self.namespaces = []
    self.typeids    = []
    self.files      = {}
    self.typedefs   = []
    self.basictypes = []
    self.methods    = []
    self.functions  = []
    self.enums      = []
    self.variables  = []
    self.vtables    = {}
    self.hfile      = os.path.normpath(hfile).replace(os.sep,'/')
    self.pool       = opts.get('pool',False)
    self.quiet      = opts.get('quiet',False)
    self.xref       = {}
    self.xrefinv    = {}
    self.cppClassSelect    = {}
    self.cppVariableSelect = {}
    self.cppEnumSelect     = {}
    self.cppFunctionSelect = {}
    self.last_id    = ''
    self.transtable = string.maketrans('<>&*,: ().$', '__rp__s___d')
    self.ignoremeth = ('rbegin', 'rend', '_Eq','_Lt', 'value_comp')
    self.x_id       = iter(xrange(sys.maxint))
    self.errors     = 0
    self.warnings   = 0
    self.comments           = opts.get('comments', False)
    self.no_membertypedefs  = opts.get('no_membertypedefs', False)
    self.generated_shadow_classes = []
    self.selectionname      = 'ROOT::Reflex::Selection'
    self.unnamedNamespaces = []
    # The next is to avoid a known problem with gccxml that it generates a
    # references to id equal '_0' which is not defined anywhere
    self.xref['_0'] = {'elem':'Unknown', 'attrs':{'id':'_0','name':''}, 'subelems':[]}
#----------------------------------------------------------------------------------
  def start_element(self, name, attrs):
    if 'id' in attrs :
      self.xref[attrs['id']] = {'elem':name, 'attrs':attrs, 'subelems':[]}
      self.last_id = attrs['id']
    if name in ('EnumValue','Argument') :
      self.xref[self.last_id]['subelems'].append(attrs)
    elif name in ('Base',) :
      if not 'bases' in self.xref[self.last_id] : self.xref[self.last_id]['bases'] = []
      self.xref[self.last_id]['bases'].append(attrs)       
    elif name in ('Class','Struct') :
      self.classes.append(attrs)
    elif name in ('Function',) :
      self.functions.append(attrs)
    elif name in ('Enumeration',) :
      self.enums.append(attrs)
    elif name in ('Variable',) :
      self.variables.append(attrs)
    elif name in ('OperatorFunction',) :
      attrs['operator'] = 'true'
      self.functions.append(attrs)
    elif name in ('Constructor','Method','OperatorMethod') :
      if 'name' in attrs and attrs['name'][0:3] != '_ZT' :
        self.methods.append(attrs)
    elif name == 'Namespace' :
      self.namespaces.append(attrs)
    elif name == 'File' :
      self.files[attrs['id']] = {'name':attrs['name']}
    elif name == 'Typedef' :
      self.typedefs.append(attrs)
    elif name == 'Variable' :
      if 'name' in attrs and attrs['name'][0:4] == '_ZTV':
        if 'context' in attrs : self.vtables[attrs['context']] = attrs
    elif name == 'FundamentalType' :
      self.basictypes.append(normalizeFragment(attrs['name']))
#----------------------------------------------------------------------------------
  def findUnnamedNamespace(self):
    for ns in self.namespaces:
      if ns['name'].find('.') != -1:
        self.unnamedNamespaces.append(ns['id'])
#----------------------------------------------------------------------------------
  def collectCppSelections(self, ns) :
    for m in ns.get('members').split():
      xm = self.xref[m]
      xelem = xm['elem']
      if xelem in ('Namespace',) :
        self.collectCppSelections(xm['attrs'])
      cname = self.genTypeName(m)
      if xelem in ('Class','Struct'):
        #self.xrefinv[cname] = m
        self.cppClassSelect[cname[len(self.selectionname)+2:]] = m
      if xelem in ('Variable',):
        self.cppVariableSelect[cname[len(self.selectionname)+2:]] = m
      if xelem in ('Enumeration',):
        self.cppEnumSelect[cname[len(self.selectionname)+2:]] = m
      if xelem in ('Function',):
        self.cppFunctionSelect[cname[len(self.selectionname)+2:]] = m
#----------------------------------------------------------------------------------
  def parse(self, file) :
    p = xml.parsers.expat.ParserCreate()
    p.StartElementHandler = self.start_element
    f = open(file)
    p.ParseFile(f)
    f.close()
    for n in self.namespaces:
      if self.genTypeName(n['id']) == self.selectionname : self.collectCppSelections(n)
    self.tryCppSelections()
    self.findUnnamedNamespace()
#----------------------------------------------------------------------------------
  def tryCppSelections(self):
    for c in self.classes:
      id = self.cppClassSelect.get(self.genTypeName(c['id'],alltempl=True))
      if (id != None) :
        selection = {'id' : id}
        self.add_template_defaults (c, selection)
        self.notice_transient (c, selection)
        self.notice_autoselect (c, selection)
    for v in self.variables:
      id = self.cppVariableSelect.get(self.genTypeName(v['id']))
      if (id != None):
        if v.has_key('extra') : v['extra']['autoselect'] = 'true'
        else                  : v['extra'] = {'autoselect':'true'}
    for e in self.enums:
      id = self.cppEnumSelect.get(self.genTypeName(e['id']))
      if (id != None):
        if e.has_key('extra') : e['extra']['autoselect'] = 'true'
        else                  : e['extra'] = {'autoselect':'true'}
    for f in self.functions:
      id = self.cppFunctionSelect.get(self.genTypeName(f['id']))
      #fixme: check if signature (incl. return type?) is the same
      if (id != None):
        if f.has_key('extra') : f['extra']['autoselect'] = 'true'
        else                  : f['extra'] = {'autoselect':'true'}
    return
#----------------------------------------------------------------------------------
  def notice_transient (self, c, selection):
    transient_fields = []
    for f in self.get_fields (selection):
      tid = f['type']
      tname = self.genTypeName (tid)
      if tname.startswith (self.selectionname+"::TRANSIENT"):
	transient_fields.append (f['name'])

    if transient_fields:
      for f in self.get_fields (c):
        fname = f['name']
        if fname in transient_fields:
	  if f.has_key('extra') : f['extra']['transient'] = 'true'
	  else                  : f['extra'] = {'transient':'true'}
          transient_fields.remove (fname)

    if transient_fields:
      print "--->> genreflex: WARNING: Transient fields declared in selection " +\
            "not present in class:", \
            self.xref[selection['id']]['attrs']['name'], transient_fields
      self.warnings += 1
    return
#----------------------------------------------------------------------------------
  def notice_autoselect (self, c, selection):
    self_autoselect = 1
    for f in self.get_fields (selection):
      tid = f['type']
      tname = self.genTypeName (tid)
      if tname.startswith( self.selectionname+'::NO_SELF_AUTOSELECT'): self_autoselect = 0
      if tname.startswith (self.selectionname+'::AUTOSELECT'):
        if 'members' in c:
          for mnum in c['members'].split():
            m = self.xref[mnum]
            if 'name' in m['attrs'] and m['attrs']['name'] == f['name']:
              if m['elem'] == 'Field':
                fattrs = self.xref[m['attrs']['type']]['attrs']
                if fattrs.has_key('extra') : fattrs['extra']['autoselect'] = 'true'
                else                       : fattrs['extra'] = {'autoselect':'true'}
              else :
                print '--->> genreflex: WARNING: AUTOSELECT selection functionality for %s not implemented yet' % m['elem']
                self.warnings += 1
    if self_autoselect :
      attrs = self.xref[c['id']]['attrs']
      if attrs.has_key('extra') : attrs['extra']['autoselect'] = 'true'
      else                      : attrs['extra'] = {'autoselect':'true'}
    return
#----------------------------------------------------------------------------------
  def get_fields (self, c):
    xref = self.xref
    cid = c['id']
    attrs = xref[cid]['attrs']
    return [xref[m]['attrs']
            for m in attrs.get('members', '').split()
            if xref[m]['elem'] == 'Field']
#----------------------------------------------------------------------------------
  def get_member (self, c, name):
    xref = self.xref
    cid = c['id']
    attrs = self.xref[cid]['attrs']
    for m in attrs.get('members', '').split():
      if xref[m]['attrs']['name'] == name:
        return m
    return None
#----------------------------------------------------------------------------------  
  def has_typedef (self, c, name):
    cid = c['id']
    attrs = self.xref[cid]['attrs']
    for m in attrs.get('members', '').split():
      if (self.xref[m]['elem'] == 'Typedef' and
          self.xref[m]['attrs']['name'] == name):
        return self.xref[m]['attrs']['type']
    return None
#----------------------------------------------------------------------------------
  def selclasses(self, sel, deep) :
    selec = []
    if sel :
      self.selector = sel  # remember the selector
      for c in self.classes :
        if 'incomplete' in c : continue
        # this check fixes a bug in gccxml 0.6.0_patch3 which sometimes generates incomplete definitions 
        # of classes (without members). Every version after 0.6.0_patch3 is tested and fixes this bug
        if not c.has_key('members') : continue
        match = self.selector.matchclass( self.genTypeName(c['id']), self.files[c['file']]['name'])
        if match[0] and not match[1] :
          c['extra'] = match[0]
          selec.append(c)
      for t in self.typedefs:
        match = self.selector.matchclass( self.genTypeName(t['id']), self.files[t['file']]['name'])
        if match[0] and not match[1] :
          c = self.xref[t['type']]
          while c['elem'] == 'Typedef': c = self.xref[c['attrs']['type']]
          if c['elem'] in ('Class','Struct'):
            self.genTypeID(t['id'])
            catt = c['attrs']
            catt['extra'] = match[0]
            if catt not in selec : selec.append(catt)
      return self.autosel (selec)
    else : self.selector = None
    local = filter(self.filefilter, self.classes)
    typed = self.typedefclasses()
    templ = self.tmplclasses(local)
    if deep :
      types = [] 
      for c in local : self.getdependent(c['id'], types)
      for c in typed : self.getdependent(c['id'], types)
      for c in templ : self.getdependent(c['id'], types)
      classes =  map( lambda t : self.xref[t]['attrs'], types)
    else :
      classes =  clean( local + typed + templ )
    # Filter STL implementation specific classes
    classes =  filter( lambda c: self.genTypeName(c['id'])[:6] != 'std::_' ,classes)
    classes =  filter( lambda c: c['name'][:2] != '._' ,classes)  # unamed structs and unions
    return self.autosel( classes )
 #----------------------------------------------------------------------------------
  def autosel(self, classes):
    types = []
    for c in self.classes:
      self.getdependent(c['id'], types)
    for t in types:
      c = self.xref[t]['attrs']
      if 'extra' in c and c['extra'].get('autoselect') and c not in classes:
        classes.append (c)
    return classes  
#----------------------------------------------------------------------------------
  def selfunctions(self, sel) :
    selec = []
    self.selector = sel  # remember the selector
    if self.selector :
      for f in self.functions :
        funcname = self.genTypeName(f['id'])
        if self.selector.selfunction( funcname ) and not self.selector.excfunction( funcname ) :
          selec.append(f)
        if 'extra' in f and f['extra'].get('autoselect') and f not in selec:
          selec.append(f)
    return selec
#----------------------------------------------------------------------------------
  def selenums(self, sel) :
    selec = []
    self.selector = sel  # remember the selector
    if self.selector :
      for e in self.enums :
        ename = self.genTypeName(e['id'])
        if self.selector.selenum( ename ) and not self.selector.excenum( ename ) :
          selec.append(e)
        if 'extra' in e and e['extra'].get('autoselect') and e not in selec:
          selec.append(e)
    return selec
#---------------------------------------------------------------------------------
  def selvariables(self, sel) :
    selec = []
    self.selector = sel  # remember the selector
    if self.selector :
      for v in self.variables :
        varname = self.genTypeName(v['id'])
        if self.selector.selvariable( varname ) and not self.selector.excvariable( varname ) :
          selec.append(v)
        if 'extra' in v and v['extra'].get('autoselect') and v not in selec:
          selec.append(v)
    return selec
#----------------------------------------------------------------------------------
  def getdependent(self, cid, types ) :
    elem  = self.xref[cid]['elem']
    attrs = self.xref[cid]['attrs']
    if elem in ('Typedef', 'ArrayType', 'PointerType','ReferenceType' ): 
      self.getdependent(attrs['type'], types)
    elif elem in ('Class','Struct') :
      if 'incomplete' in attrs : return
      if attrs['id'] not in types : 
        types.append(attrs['id'])
        if 'members' in attrs :
          for m in attrs['members'].split() :
            if self.xref[m]['elem'] == 'Field' :
              type = self.xref[m]['attrs']['type']
              self.getdependent(type, types)
        if 'bases' in attrs :
          for b in attrs['bases'].split() :
            if b[:10] == 'protected:' : b = b[10:]
            if b[:8]  == 'private:'   : b = b[8:]
            self.getdependent(b, types)
#----------------------------------------------------------------------------------
  def generate(self, file, selclasses, selfunctions, selenums, selvariables, cppinfo) :
    names = []
    f = open(file,'w') 
    f.write(self.genHeaders(cppinfo))
    f_buffer = ''
    f_shadow =  '\n// Shadow classes to obtain the data member offsets \n'
    f_shadow += 'namespace __shadow__ {\n'
    for c in selclasses :
      if 'incomplete' not in c :
        cname = self.genTypeName(c['id'])
        if not self.quiet : print  'class '+ cname
        names.append(cname)
        self.completeClass( c )
        self.enhanceClass( c )
        scons, stubs   = self.genClassDict( c )
        f_buffer += stubs
        f_buffer += scons
        f_shadow += self.genClassShadow(c)
    f_shadow += '}\n\n'
    f_buffer += self.genFunctionsStubs( selfunctions )
    f_buffer += self.genInstantiateDict(selclasses, selfunctions, selenums, selvariables)
    f.write('namespace {\n')
    f.write(self.genNamespaces(selclasses + selfunctions + selenums + selvariables))
    f.write(self.genAllTypes())
    f.write('} // unnamed namespace\n')
    f.write(f_shadow)
    f.write('namespace {\n')
    f.write(f_buffer)
    f.write('} // unnamed namespace\n')
    f.close()
    return names, self.warnings, self.errors
#----------------------------------------------------------------------------------
  def add_template_defaults (self, c, selection):
    tlist = []
    for f in self.get_fields (selection):
        tid = f['type']
        tname = self.genTypeName (tid)
        if tname.startswith (self.selectionname+"::TEMPLATE_DEFAULTS"):
          tid = {'id': tid}
          nodefault_tid = self.has_typedef (tid, 'nodefault')
          i = 1
          while 1:
            arg = self.has_typedef (tid, 't%d' % i)
            if not arg:
              break
            if arg == nodefault_tid:
              tlist.append ('=')
            else:
              tlist.append (self.genTypeName (arg))
            i += 1
    if tlist:
      name = self.xref[c['id']]['attrs']['name']
      i = name.find ('<')
      if i>=0 : name = name[:i]
      stldeftab[name] = tuple (tlist)
    return
#----------------------------------------------------------------------------------
  def isUnnamedType(self, name) :
    if name and (name.find('.') != -1 or name.find('$') != -1): return 1
    else                                            : return 0
#----------------------------------------------------------------------------------
  def filefilter(self, attrs):
    if self.genTypeName(attrs['id'])[:len(self.selectionname)] == self.selectionname : return 0
    fileid = attrs['file']
    if self.files[fileid]['name'] == self.hfile : return 1
    else : return 0
#----------------------------------------------------------------------------------
  def memberfilter( self, id ) :
    elem  = self.xref[id]['elem']
    attrs = self.xref[id]['attrs']
    args  = self.xref[id]['subelems']
    if 'name' in attrs :
       if attrs['name'] in self.ignoremeth : return 0
    #----Filter any method and operator for POOL -----
    if self.pool :
      if elem in ('OperatorMethod','Converter') : return 0
      elif elem in ('Method',) :
        if attrs['name'] not in ('at','size','clear','resize') : return 0
      elif elem in ('Constructor',) :
        if len(args) > 1 : return 0
        elif len(args) == 1 :
          if self.genTypeName(args[0]['type']) != 'const '+self.genTypeName(attrs['context'])+'&' : return 0
    #----Filter any non public method
    if 'access' in attrs :  # assumes that the default is "public"
      if elem in ('Constructor','Destructor','Method','OperatorMethod','Converter') : return 0
    #----Filter any copy constructor with a private copy constructor in any base
    if elem == 'Constructor' and len(args) == 1 and 'name' in args[0] and args[0]['name'] == '_ctor_arg' :
      if self.isConstructorPrivate(attrs['context']) : return 0
    #----Filter any constructor for pure abstract classes
    if 'context' in attrs :
      if 'abstract' in self.xref[attrs['context']]['attrs'] : 
        if elem in ('Constructor',) : return 0
    #----Filter using the exclusion list in the selection file
    if self.selector and 'name' in attrs and  elem in ('Constructor','Destructor','Method','OperatorMethod','Converter') :
      if self.selector.excmethod(self.genTypeName(attrs['context']), attrs['name'] ) : return 0
    return 1
#----------------------------------------------------------------------------------
  def tmplclasses(self, local):
    result = []
    for c in self.classes :
      name = c['name']
      if name.find('<') == -1 : continue
      temp = name[name.find('<')+1:name.rfind('>')]
      for lc in local :
        if temp.find(lc['name']) != -1 : result.append(c)
    return result
#----------------------------------------------------------------------------------
  def typedefclasses(self):
    result = []
    for t in self.typedefs :
      fileid = t['location']
      fileid = fileid[:fileid.index(':')]
      if self.xref[fileid]['attrs']['name'] == self.hfile : 
        if self.xref[t['type']]['elem'] in ('Class','Struct') :
          result.append(self.xref[t['type']]['attrs'])
    return result
#----------------------------------------------------------------------------------
  def isConstructorPrivate(self, id ) :
    attrs = self.xref[id]['attrs']
    if 'members' in attrs : 
       for m in attrs['members'].split() :
         elem = self.xref[m]['elem']
         attr = self.xref[m]['attrs']
         args = self.xref[m]['subelems']
         if elem == 'Constructor' and len(args) == 1 :
           if self.genTypeName(args[0]['type']) == 'const '+self.genTypeName(attr['context'])+'&' :
             if 'access' in attr and attr['access'] == 'private' : return True
    if 'bases' in attrs :
       for b in attrs['bases'].split() :
         if b[:10] == 'protected:' : b = b[10:]
         if b[:8]  == 'private:'   : b = b[8:]
         if self.isConstructorPrivate(b) : return True
    return False
#----------------------------------------------------------------------------------
  def isDestructorNonPublic(self, id ) :
    attrs = self.xref[id]['attrs']
    if 'members' in attrs : 
       for m in attrs['members'].split() :
         elem = self.xref[m]['elem']
         attr = self.xref[m]['attrs']
         if elem == 'Destructor' :
           if attr.get('access') in ('private','protected') : return True
           else : return False
    if 'bases' in attrs :
       for b in attrs['bases'].split() :
         if b[:10] == 'protected:' : b = b[10:]
         if b[:8]  == 'private:'   : b = b[8:]
         if self.isDestructorNonPublic(b) : return True
    return False
#----------------------------------------------------------------------------------
  def isClassVirtual(self, attrs ) :
    if 'members' in attrs : 
       for m in attrs['members'].split() :
         elem = self.xref[m]['elem']
         attr = self.xref[m]['attrs']
         if elem in ('Destructor','Method') :
             if 'virtual' in attr : return True
    if 'bases' in attrs :
       for b in attrs['bases'].split() :
         if b[:10] == 'protected:' : b = b[10:]
         if b[:8]  == 'private:'   : b = b[8:]
         if self.isClassVirtual(self.xref[b]['attrs']) : return True
    return False
#----------------------------------------------------------------------------------
  def isClassPublic(self, id ) :
    attrs = self.xref[id]['attrs']
    if 'access' in attrs : return False
    elif attrs['name'][-1] == '>' :
      args = getTemplateArgs(attrs['name'])
      for a in args :
        while a[-1] in ('*','&') : a = a[:-1]
        a = a.replace(', ',',')
        if a in self.xrefinv :
          if not self.isClassPublic(self.xrefinv[a]) : return False
        else :
          print '#%s#'% a, ' is not found in the table' 
    return True
#----------------------------------------------------------------------------------
  def genHeaders(self, gccxmlinfo):
    c =  '// Generated at %s. Do not modify it\n\n' % time.ctime(time.time())
    if (gccxmlinfo) : c += '/*\n%s*/\n\n' % gccxmlinfo
    c += '#ifdef _WIN32\n'
    c += '#pragma warning ( disable : 4786 )\n'
    c += '#endif\n'
    c += '#include "%s"\n' % self.hfile
    c += '#include "Reflex/Builder/ReflexBuilder.h"\n'
    c += '#include <typeinfo>\n'
    c += 'using namespace ROOT::Reflex;\n\n'
    return c
#----------------------------------------------------------------------------------
  def genInstantiateDict( self, selclasses, selfunctions, selenums, selvariables) :
    c = 'namespace {\n  struct Dictionaries {\n    Dictionaries() {\n'
    for attrs in selclasses :
      if 'incomplete' not in attrs : 
        clf = self.genTypeName(attrs['id'], colon=True)
        clt = string.translate(str(clf), self.transtable)
        c += '      %s_dict(); \n' % (clt)
    c += self.genFunctions(selfunctions)
    c += self.genEnums(selenums)
    c += self.genVariables(selvariables)
    c += '    }\n    ~Dictionaries() {\n'
    for attrs in selclasses :
      if 'incomplete' not in attrs : 
        cls = self.genTypeName(attrs['id'])
        c += '      %s.Unload(); // class %s \n' % (self.genTypeID(attrs['id']), cls)
    c += '    }\n  };\n'
    c += '  static Dictionaries instance;\n}\n'
    return c
#---------------------------------------------------------------------------------
  def genClassDict(self, attrs):
    members, bases = [], []
    cl  = attrs['name']
    clf = self.genTypeName(attrs['id'],colon=True)
    cls = self.genTypeName(attrs['id'])
    clt = string.translate(str(clf), self.transtable)
    bases = self.getBases( attrs['id'] )
    if 'members' in attrs : members = string.split(attrs['members'])
    mod = self.genModifier(attrs,None)
    mod += ' | ' + self.xref[attrs['id']]['elem'].upper()
    if attrs.has_key('abstract') : mod += ' | ABSTRACT'
    if self.vtables :
      if attrs['id'] in self.vtables : mod += ' | VIRTUAL'
    else :  # new in version 0.6.0
      if self.isClassVirtual(attrs) :  mod += ' | VIRTUAL'
    members = filter(self.memberfilter, members)  # Eliminate problematic members
    # Fill the different streams sc: constructor, ss: stub functions
    sc = '//------Dictionary for class %s -------------------------------\n' % cl
    sc += 'void %s_dict() {\n' % (clt,)
    if 'extra' in attrs and 'contid' in attrs['extra'] : 
      cid = attrs['extra']['contid'].upper()
    else :
      cid = getContainerId(clf)[0]
    notAccessibleType = self.checkAccessibleType(self.xref[attrs['id']])
    if self.isUnnamedType(clf) : 
      sc += '  ClassBuilder("%s", typeid(Unnamed%s), sizeof(%s), %s)' % ( cls, self.xref[attrs['id']]['elem'], '__shadow__::'+ string.translate(str(clf),self.transtable),mod )
    elif notAccessibleType :
      sc += '  ClassBuilder("%s", typeid(%s%s), sizeof(%s), %s)' % ( cls, self.xref[notAccessibleType]['attrs']['access'].title(), self.xref[attrs['id']]['elem'], '__shadow__::'+ string.translate(str(clf),self.transtable),mod )
    else :
      sc += '  ClassBuilder("%s", typeid(%s), sizeof(%s), %s)' % (cls, cls, cls, mod)
    if 'extra' in attrs :
      for pname, pval in attrs['extra'].items() :
        if pname not in ('name','pattern','n_name','file_name','file_pattern') :
          if pname == 'id' : pname = 'ClassID'
          sc += '\n  .AddProperty("%s", "%s")' % (pname, pval)
    for b in bases :
      sc += '\n' + self.genBaseClassBuild( clf, b )
    for m in members :
      funcname = 'gen'+self.xref[m]['elem']+'Build'
      if funcname in dir(self) :
        line = self.__class__.__dict__[funcname](self, self.xref[m]['attrs'], self.xref[m]['subelems'])
        if line : sc += '\n' + line 
    sc += ';\n}\n\n'
    ss = ''
    if not self.isUnnamedType(clf) and not notAccessibleType:
      ss = '//------Stub functions for class %s -------------------------------\n' % cl
      for m in members :
        funcname = 'gen'+self.xref[m]['elem']+'Def'
        if funcname in dir(self) :
          ss += self.__class__.__dict__[funcname](self, self.xref[m]['attrs'], self.xref[m]['subelems']) + '\n'
    return sc, ss
#----------------------------------------------------------------------------------
  def checkAccessibleType( self, type ):
    while type['elem'] in ('PointerType','Typedef') : type = self.xref[type['attrs']['type']]
    attrs = type['attrs']
    if 'access' in attrs and attrs['access'] in ('private','protected') : return attrs['id']
    if 'context' in attrs and self.checkAccessibleType(self.xref[attrs['context']]) : return attrs['id']
    return 0
#----------------------------------------------------------------------------------
  def genClassShadow(self, attrs, inner = 0 ) :
    inner_shadows = {}
    bases = self.getBases( attrs['id'] )
    cls = self.genTypeName(attrs['id'],const=True,colon=True)
    clt = string.translate(str(cls), self.transtable)
    if self.isUnnamedType(cls) and inner : clt = ''
    xtyp = self.xref[attrs['id']]
    typ = xtyp['elem'].lower()
    indent = inner * 2 * ' '
    if typ == 'enumeration' :
      c = indent + 'enum %s {};\n' % clt
    else:
      if not bases : 
        c = indent + '%s %s {\n%s  public:\n' % (typ, clt, indent)
      else :
        c = indent + '%s %s : ' % (typ, clt)
        for b in bases :
          if b.get('virtual','') == '1' : acc = 'virtual ' + b['access']
          else                          : acc = b['access']
	  bname = self.genTypeName(b['type'],colon=True)
	  if self.xref[b['type']]['attrs'].get('access') in ('private','protected'):
            bname = string.translate(str(bname),self.transtable)
          c += indent + '%s %s' % ( acc , bname )
          if b is not bases[-1] : c += ', ' 
        c += indent + ' {\n' + indent +'  public:\n'
      if clt: # and not self.checkAccessibleType(xtyp):
        c += indent + '  %s();\n' % (clt)
        if self.isClassVirtual( attrs ) :
          c += indent + '  virtual ~%s() throw();\n' % ( clt )
      members = attrs.get('members','')
      memList = members.split()
      for m in memList :
        member = self.xref[m]
        if member['elem'] in ('Class','Struct','Union','Enumeration') \
           and member['attrs'].get('access') in ('private','protected') \
           and not self.isUnnamedType(member['attrs'].get('name')):
          cmem = self.genTypeName(member['attrs']['id'],const=True,colon=True)
          if cmem != cls and cmem not in inner_shadows :
            inner_shadows[cmem] = string.translate(str(cmem), self.transtable)
            c += self.genClassShadow(member['attrs'], inner + 1)
      for m in memList :
        member = self.xref[m]
        if member['elem'] in ('Field',) :
          a = member['attrs']
          t = self.genTypeName(a['type'],colon=True,const=True)
          #---- Check if a type and a member with the same name exist in the same scope
          mTypeElem = self.xref[a['type']]['elem']
          if mTypeElem in ('Class','Struct'):
            mTypeName = self.xref[a['type']]['attrs']['name']
            mTypeId = a['type']
            for el in self.xref[self.xref[a['type']]['attrs']['context']]['attrs'].get('members').split():
              if self.xref[el]['attrs'].get('name') == mTypeName and mTypeId != el :
                t = mTypeElem.lower() + ' ' + t[2:]
                break
          #---- Check for non public types------------------------
          noPublicType = self.checkAccessibleType(self.xref[a['type']])
          if ( noPublicType and not self.isUnnamedType(self.xref[a['type']]['attrs'].get('name'))):
            noPubTypeAttrs = self.xref[noPublicType]['attrs']
            cmem = self.genTypeName(noPubTypeAttrs['id'],const=True,colon=True)
            if cmem != cls and cmem not in inner_shadows :
              inner_shadows[cmem] = string.translate(str(cmem), self.transtable)
              c += self.genClassShadow(noPubTypeAttrs, inner + 1)
          #---- translate the type with the inner shadow type-----  
          ikeys = inner_shadows.keys()
          ikeys.sort(lambda x,y : len(y) - len(x))
          for ikey in ikeys :      
            if   t.find(ikey) == 0      : t = t.replace(ikey, inner_shadows[ikey])     # change current class by shadow name 
            elif t.find(ikey[2:]) != -1 : t = t.replace(ikey[2:], inner_shadows[ikey]) # idem without leading ::
          mType = self.xref[a.get('type')]
          if mType and self.isUnnamedType(mType['attrs'].get('name')) :
            t = self.genClassShadow(mType['attrs'], inner+1)[:-2]
          if t[-1] == ']'         : c += indent + '  %s %s;\n' % ( t[:t.find('[')], a['name']+t[t.find('['):] )
          elif t.find(')(') != -1 and t.find('<') == -1 : c += indent + '  %s;\n' % ( t.replace(')(', ' %s)('%a['name']))
          else                    : c += indent + '  %s %s;\n' % ( t, a['name'] )
      c += indent + '};\n'
    return c    
#----------------------------------------------------------------------------------
  def genTypedefBuild(self, attrs, childs) :
    if self.no_membertypedefs : return ''
    s = ''
    s += '  .AddTypedef(%s, "%s::%s")' % ( self.genTypeID(attrs['type']), self.genTypeName(attrs['context']), attrs['name']) 
    return s  
#----------------------------------------------------------------------------------
  def genEnumerationBuild(self, attrs, childs):
    s = ''
    name = self.genTypeName(attrs['id']) 
    values = ''
    for child in childs : values += child['name'] + '=' + child['init'] +';'
    values = values[:-1]
    if self.isUnnamedType(name) :
      s += '  .AddEnum("%s", "%s", &typeid(ROOT::Reflex::UnnamedEnum))' % (name[name.rfind('::')+3:], values) 
    else :
      if attrs.get('access') in ('protected','private'):
        s += '  .AddEnum("%s", "%s", &typeid(ROOT::Reflex::UnknownType))' % (name, values)        
      else:
        s += '  .AddEnum("%s", "%s", &typeid(%s))' % (name, values, name)
    return s 
#----------------------------------------------------------------------------------
  def genScopeName(self, attrs, enum=False, const=False, colon=False) :
    s = ''
    if 'context' in attrs :
      ctxt = attrs['context']
      while ctxt in self.unnamedNamespaces: ctxt = self.xref[ctxt]['attrs']['context']
      ns = self.genTypeName(ctxt, enum, const, colon)
      if ns : s = ns + '::'
      elif colon  : s = '::'
    return s
#----------------------------------------------------------------------------------
  def genTypeName(self, id, enum=False, const=False, colon=False, alltempl=False) :
    if id[-1] in ['c','v'] :
      nid = id[:-1]
      cvdict = {'c':'const','v':'volatile'}
      prdict = {'PointerType':'*', 'ReferenceType':'&'}
      nidelem = self.xref[nid]['elem']
      if nidelem in ('PointerType','ReferenceType') :
        if const : return self.genTypeName(nid, enum, const, colon)
        else :     return self.genTypeName(nid, enum, const, colon) + ' ' + cvdict[id[-1]]
      else :
        if const : return self.genTypeName(nid, enum, const, colon)
        else     : return cvdict[id[-1]] + ' ' + self.genTypeName(nid, enum, const, colon)
    elem  = self.xref[id]['elem']
    attrs = self.xref[id]['attrs']
    s = self.genScopeName(attrs, enum, const, colon)
    if elem == 'Namespace' :
      if attrs['name'] != '::' : s += attrs['name']
    elif elem == 'PointerType' :
      t = self.genTypeName(attrs['type'],enum, const, colon)
      if   t[-1] == ')' or t[-7:] == ') const' or t[-10:] == ') volatile' : s += t.replace('::*)','::**)').replace('::)','::*)').replace('(*)', '(**)').replace('()','(*)')
      elif t[-1] == ']' or t[-7:] == ') const' or t[-10:] == ') volatile' : s += t[:t.find('[')] + '(*)' + t[t.find('['):]
      else              : s += t + '*'   
    elif elem == 'ReferenceType' :
      s += self.genTypeName(attrs['type'],enum, const, colon)+'&'
    elif elem in ('FunctionType','MethodType') :
      s = self.genTypeName(attrs['returns'], enum, const, colon)
      if elem == 'MethodType' : 
        s += '('+ self.genTypeName(attrs['basetype'], enum, const, colon) + '::)('
      else :
        s += '()('
      args = self.xref[id]['subelems']
      if args :
        for a in range(len(args)) :
          s += self.genTypeName(args[a]['type'])
          if a < len(args)-1 : s += ', '
        s += ')'
      else :
        s += 'void)'
      if (attrs.get('const') == '1') : s += ' const'
      if (attrs.get('volatile') == '1') : s += ' volatile'
    elif elem == 'ArrayType' :
      arr = '[%s]' % str(int(attrs['max'])+1)
      typ = self.genTypeName(attrs['type'], enum, const, colon)
      if typ[-1] == ']' :
        pos = typ.find('[')
        s += typ[:pos] + arr + typ[pos:]
      else:
        s += typ + arr
    elif elem == 'Unimplemented' :
      s += attrs['tree_code_name']
    elif elem == 'Enumeration' :
      if enum : s = 'int'           # Replace "enum type" by "int"
      else :    s += attrs['name']  # FIXME: Not always true  
    elif elem == 'Typedef' :
      s = self.genScopeName(attrs, enum, const, colon)
      s += attrs['name']
    elif elem in ('Function', 'OperatorFunction') :
      if 'name' in attrs : s += attrs['name']
      else : pass
    elif elem == 'OffsetType' :
      s += self.genTypeName(attrs['type'], enum, const, colon) + ' '
      s += self.genTypeName(attrs['basetype'], enum, const, colon) + '::'  
    else :
      if 'name' in attrs : s += attrs['name']
      s = normalizeClass(s,alltempl)                   # Normalize STL class names, primitives, etc.
    return s
#----------------------------------------------------------------------------------
  def genTypeID(self, id ) :
    if id[-1] in ('c','v') :
      self.genTypeID(id[:-1])
    else : 
      elem  = self.xref[id]['elem']
      attrs = self.xref[id]['attrs']
      if elem in ('PointerType', 'ReferenceType', 'ArrayType', 'Typedef') :
        self.genTypeID(attrs['type'])
      elif elem in ('FunctionType', 'MethodType') :
        if 'returns' in attrs : self.genTypeID(attrs['returns'])
        args = self.xref[id]['subelems']
        for a in args : self.genTypeID(a['type'])
      elif elem in ('OperatorMethod', 'Method', 'Constructor', 'Converter', 'Destructor', 
                    'Function', 'OperatorFunction' ) :
        if 'returns' in attrs : c = 'FunctionTypeBuilder(' + self.genTypeID(attrs['returns'])
        else                  : c = 'FunctionTypeBuilder(type_void'
        args = self.xref[id]['subelems']
        for a in args : c += ', '+ self.genTypeID(a['type'])
        c += ')'
        return c
      elif elem in ('Variable',) :
        self.genTypeID(attrs['type'])
      else :
        pass
    # Add this type in the list of types...
    if id not in self.typeids : self.typeids.append(id)
    return 'type'+id
#----------------------------------------------------------------------------------
  def genAllTypes(self) :
    c = '  Type type_void = TypeBuilder("void");\n'
    for id in self.typeids :      
      c += '  Type type%s = ' % id
      if id[-1] == 'c':
        c += 'ConstBuilder(type'+id[:-1]+');\n'
      elif id[-1] == 'v':
        c += 'VolatileBuilder(type'+id[:-1]+');\n'
      else : 
        elem  = self.xref[id]['elem']
        attrs = self.xref[id]['attrs']
        if elem == 'PointerType' :
          c += 'PointerBuilder(type'+attrs['type']+');\n'
        elif elem == 'ReferenceType' :
          c += 'ReferenceBuilder(type'+attrs['type']+');\n'
        elif elem == 'ArrayType' :
          mx = attrs['max']
          # check if array is bound (max='fff...' for unbound arrays)
          if mx.isdigit() : len = str(int(mx)+1)
          else            : len = '0' 
          c += 'ArrayBuilder(type'+attrs['type']+', '+ len +');\n'
        elif elem == 'Typedef' :
          sc = self.genTypeName(attrs['context'])
          if sc : sc += '::'
          c += 'TypedefTypeBuilder("'+sc+attrs['name']+'", type'+ attrs['type']+');\n'
        elif elem == 'OffsetType' :
          c += 'TypeBuilder("%s");\n' % self.genTypeName(attrs['id'])
        elif elem == 'FunctionType' :
          if 'returns' in attrs : c += 'FunctionTypeBuilder(type'+attrs['returns']
          else                  : c += 'FunctionTypeBuilder(type_void'
          args = self.xref[id]['subelems']
          for a in args : c += ', type'+ a['type']
          c += ');\n'
        elif elem == 'MethodType' :
          c += 'TypeBuilder("%s");\n' % self.genTypeName(attrs['id'])
        elif elem in ('OperatorMethod', 'Method', 'Constructor', 'Converter', 'Destructor',
                      'Function', 'OperatorFunction') :
          pass
        elif elem == 'Enumeration' :
          sc = self.genTypeName(attrs['context'])
          if sc : sc += '::'
          # items = self.xref[id]['subelems']
          # values = string.join([ item['name'] + '=' + item['init'] for item in items],';"\n  "')          
          #c += 'EnumTypeBuilder("' + sc + attrs['name'] + '", "' + values + '");\n'
          c += 'EnumTypeBuilder("' + sc + attrs['name'] + '");\n'
        else :
         name = ''
         if 'context' in attrs :
           ns = self.genTypeName(attrs['context'])
           if ns : name += ns + '::'
         if 'name' in attrs :
           name += attrs['name']
         name = normalizeClass(name,False)
         c += 'TypeBuilder("'+name+'");\n'
    return c 
#----------------------------------------------------------------------------------
  def genNamespaces(self, selected ) :
    used_context = []
    s = ''
    for c in selected :
      if 'incomplete' not in c : used_context.append(c['context'])
    idx = 0
    for ns in self.namespaces :
      if ns['id'] in used_context and ns['name'] != '::' :
        s += '  NamespaceBuilder nsb%d( "%s" );\n' % (idx, self.genTypeName(ns['id']))
        idx += 1
    return s
#----------------------------------------------------------------------------------
  def genFunctionsStubs(self, selfunctions) :
    s = ''
    for f in selfunctions :
      id   = f['id']
      name = self.genTypeName(id)
      if 'operator' in f : name = 'operator '+name
      self.genTypeID(id)
      args = self.xref[id]['subelems']
      returns  = self.genTypeName(f['returns'], enum=True, const=True)
      if not self.quiet : print  'function '+ name
      s += 'static void* '
      if len(args) :
        s +=  'function%s( void*, const std::vector<void*>& arg, void*)\n{\n' % id 
      else :
        s +=  'function%s( void*, const std::vector<void*>&, void*)\n{\n' % id
      ndarg = self.getDefaultArgs(args)
      narg  = len(args)
      if ndarg : iden = '  '
      else     : iden = ''
      if returns != 'void' and returns in self.basictypes :
        s += '  static %s ret;\n' % returns
      for n in range(narg-ndarg, narg+1) :
        if ndarg :
          if n == narg-ndarg :  s += '  if ( arg.size() == %d ) {\n' % n
          else               :  s += '  else if ( arg.size() == %d ) { \n' % n
        if returns == 'void' :
          first = iden + '  %s(' % ( name, )
          s += first + self.genMCOArgs(args, n, len(first)) + ');\n'
          s += iden + '  return 0;\n'
        else :
          if returns[-1] in ('*',')' ):
            first = iden + '  return (void*)%s(' % ( name, )
            s += first + self.genMCOArgs(args, n, len(first)) + ');\n'
          elif returns[-1] == '&' :
            first = iden + '  return (void*)&%s(' % ( name, )
            s += first + self.genMCOArgs(args, n, len(first)) + ');\n'
          elif returns in self.basictypes :
            first = iden + '  ret = %s(' % ( name, )
            s += first + self.genMCOArgs(args, n, len(first)) + ');\n'
            s += iden + '  return &ret;\n'        
          else :
            first = iden + '  return new %s(%s(' % ( returns, name )
            s += first + self.genMCOArgs(args, n, len(first)) + '));\n'
        if ndarg : 
          if n != narg : s += '  }\n'
          else :
            if returns == 'void' : s += '  }\n  return 0;\n'
            else :                 s += '  }\n  return 0;\n'
      s += '}\n'
    return s  
#----------------------------------------------------------------------------------
  def genFunctions(self, selfunctions) :
    s = ''
    i = 0;
    for f in selfunctions :
      id   = f['id']
      name = self.genTypeName(id)
      if ( self.xref[id]['attrs'].has_key('mangled') ):
        mm = self.xref[id]['attrs']['mangled'][2:]
        dname = gccdemangler.demangle_name(mm)
      else :
        dname = name
      name += getTemplateArgString(dname[1])
      args = self.xref[id]['subelems']      
      if args : params  = '"'+ string.join( map(self.genParameter, args),';')+'"'
      else    : params  = '0'
      mod = self.genModifier(f, None)
      s += '      Type t%s = %s;' % (i, self.genTypeID(id))
      s += '      FunctionBuilder(t%s, "%s", function%s, 0, %s, %s);\n' % (i, name, id, params, mod)
      i += 1;
    return s
#----------------------------------------------------------------------------------
  def genEnums(self, selenums) :
    s = ''
    i = 0;
    for e in selenums :
      id   = e['id']
      cname = self.genTypeName(id, colon=True)
      name  = self.genTypeName(id)
      if not self.quiet : print 'enum ' + name
      s += '      EnumBuilder("%s",typeid(%s))' % (name, cname)
      items = self.xref[id]['subelems']
      for item in items :
        s += '\n        .AddItem("%s",%s)' % (item['name'], item['init'])
      s += ';\n'
    return s
#----------------------------------------------------------------------------------
  def genVariables(self, selvars) :
    s = ''
    i = 0;
    for v in selvars :
      id   = v['id']
      cname = self.genTypeName(id, colon=True)
      name  = self.genTypeName(id)
      mod   = self.genModifier(v, None)
      if not self.quiet : print 'variable ' + name 
      s += '      VariableBuilder("%s", %s, (size_t)&%s, %s );\n' % (name, self.genTypeID(v['type']),self.genTypeName(id), mod)
    return s
 #----------------------------------------------------------------------------------
  def countColonsForOffset(self, name) :
    prn = 0
    cnt = 0
    for c in name :
      if c == ',' and not prn : cnt += 1
      elif c == '('           : prn += 1
      elif c == ')'           : prn -= 1
      else                    : pass
    return cnt
#----------------------------------------------------------------------------------
  def genFieldBuild(self, attrs, childs):
    type   = self.genTypeName(attrs['type'], enum=False, const=False)
    cl     = self.genTypeName(attrs['context'],colon=True)
    cls    = self.genTypeName(attrs['context'])
    name = attrs['name']
    if not name :
      ftype = self.xref[attrs['type']]
      # if the member type is an unnamed union we try to take the first member of the union as name
      if ftype['elem'] == 'Union':
        firstMember = ftype['attrs']['members'].split()[0]
        if firstMember : name = self.xref[firstMember]['attrs']['name']
        else           : return ''       # then this must be an unnamed union without members
    if type[-1] == '&' :
      print '--->> genreflex: WARNING: References are not supported as data members (%s %s::%s)' % ( type, cls, name )
      self.warnings += 1
      return ''
    if 'bits' in attrs:
      print '--->> genreflex: WARNING: Bit-fields are not supported as data members (%s %s::%s:%s)' % ( type, cls, name, attrs['bits'] )
      self.warnings += 1
      return ''
    if self.selector : xattrs = self.selector.selfield( cls,name)
    else             : xattrs = None
    mod = self.genModifier(attrs,xattrs)
    shadow = '__shadow__::' + string.translate( str(cl), self.transtable)
    c = '  .AddDataMember(%s, "%s", OffsetOf(%s, %s), %s)' % (self.genTypeID(attrs['type']), name, shadow, name, mod)
    c += self.genCommentProperty(attrs)
    # Other properties
    if xattrs : 
      for pname, pval in xattrs.items() : 
        if pname not in ('name', 'transient', 'pattern') :
          c += '\n  .AddProperty("%s","%s")' % (pname, pval)     
    return c
#----------------------------------------------------------------------------------    
  def genCommentProperty(self, attrs):
    if not self.comments or 'file' not in attrs or ('artificial' in attrs and attrs['artificial'] == '1') : return '' 
    fd = self.files[attrs['file']]
    # open and read the header file if not yet done
    if 'filelines' not in fd :
      try :
        f = file(fd['name'])
        fd['filelines'] = f.readlines()
        f.close()
      except :
        return ''
    line = fd['filelines'][int(attrs['line'])-1]
    if line.find('//') == -1 : return ''
    return '\n  .AddProperty("comment","%s")' %  (line[line.index('//')+2:-1]).replace('"','\\"')
#----------------------------------------------------------------------------------
  def genArgument(self, attrs):
    c = self.genTypeName(attrs['type'], enum=True, const=False)
    return c
#----------------------------------------------------------------------------------
  def genParameter(self, attrs):
    c = ''
    if 'name' in attrs :
      c += attrs['name']
      if 'default' in attrs :
        c += '='+ attrs['default'].replace('"','\\"')
    return c
#----------------------------------------------------------------------------------
  def genModifier(self, attrs, xattrs ):
    if 'access' not in attrs            : mod = 'PUBLIC'
    elif attrs['access'] == 'private'   : mod = 'PRIVATE'
    elif attrs['access'] == 'protected' : mod = 'PROTECTED'
    else                                : mod = 'NONE'
    if 'virtual' in attrs : mod += ' | VIRTUAL'
    if 'static'  in attrs : mod += ' | STATIC'
    # Extra modifiers
    xtrans = ''
    etrans = ''
    if xattrs :
      xtrans = xattrs.get('transient')
      if xtrans : xtrans = xtrans.lower()
    if 'extra' in attrs:
      etrans = attrs['extra'].get('transient')
      if etrans : etrans = etrans.lower()
    if xtrans == 'true' or etrans == 'true' : mod += ' | TRANSIENT'
    if 'artificial' in attrs : mod += ' | ARTIFICIAL' 
    return mod
#----------------------------------------------------------------------------------
  def genMCODecl( self, type, name, attrs, args ) :
    return 'static void* %s%s(void*, const std::vector<void*>&, void*);' % (type, attrs['id'])
#----------------------------------------------------------------------------------
  def genMCOBuild(self, type, name, attrs, args):
    id       = attrs['id']
    if self.isUnnamedType(self.xref[attrs['context']]['attrs'].get('name')) or \
       self.checkAccessibleType(self.xref[attrs['context']]) : return ''
    if type == 'constructor' : returns  = 'void'
    else                     : returns  = self.genTypeName(attrs['returns'])
    mod = self.genModifier(attrs, None)
    if   type == 'constructor' : mod += ' | CONSTRUCTOR'
    elif type == 'operator' :    mod += ' | OPERATOR'
    elif type == 'converter' :   mod += ' | CONVERTER'
    if attrs.get('const')=='1' : mod += ' | CONST'
    if args : params  = '"'+ string.join( map(self.genParameter, args),';')+'"'
    else    : params  = '0'
    s = '  .AddFunctionMember(%s, "%s", %s%s, 0, %s, %s)' % (self.genTypeID(id), name, type, id, params, mod)
    s += self.genCommentProperty(attrs)
    return s
#----------------------------------------------------------------------------------
  def genMCODef(self, type, name, attrs, args):
    id       = attrs['id']
    cl       = self.genTypeName(attrs['context'],colon=True)
    clt      = string.translate(str(cl), self.transtable)
    returns  = self.genTypeName(attrs['returns'],enum=True, const=True)
    s = 'static void* '
    if len(args) :
      s +=  '%s%s( void* o, const std::vector<void*>& arg, void*)\n{\n' %( type, id )
    else :
      s +=  '%s%s( void* o, const std::vector<void*>&, void*)\n{\n' %( type, id )
    # If we construct a conversion operator to pointer to function member the name
    # will contain TDF_<attrs['id']>
    tdfname = 'TDF%s'%attrs['id']
    if name.find(tdfname) != -1 :
      s += '  typedef %s;\n'%name
      name = 'operator ' + tdfname
      returns = tdfname
    ndarg = self.getDefaultArgs(args)
    narg  = len(args)
    if ndarg : iden = '  '
    else     : iden = ''
    if returns != 'void' :
      if returns in self.basictypes or name == 'operator %s'%tdfname:
        s += '  static %s ret;\n' % returns
      elif returns.find('::*)') != -1 :
        s += '  static %s;\n' % returns.replace('::*','::* ret')
      elif returns.find('::*') != -1 :
        s += '  static %s ret;\n' % returns  
    if 'const' in attrs : cl = 'const '+ cl
    for n in range(narg-ndarg, narg+1) :
      if ndarg :
        if n == narg-ndarg :  s += '  if ( arg.size() == %d ) {\n' % n
        else               :  s += '  else if ( arg.size() == %d ) { \n' % n
      if returns == 'void' :
        first = iden + '  ((%s*)o)->%s(' % ( cl, name )
        s += first + self.genMCOArgs(args, n, len(first)) + ');\n'
        s += iden + '  return 0;\n'
      else :
        if returns[-1] in ('*',')') and returns.find('::*') == -1 :
          first = iden + '  return (void*)((%s*)o)->%s(' % ( cl, name )
          s += first + self.genMCOArgs(args, n, len(first)) + ');\n'
        elif returns[-1] == '&' :
          first = iden + '  return (void*)&((%s*)o)->%s(' % ( cl, name )
          s += first + self.genMCOArgs(args, n, len(first)) + ');\n'
        elif returns in self.basictypes or returns.find('::*') != -1 or name == 'operator '+tdfname:
          first = iden + '  ret = ((%s*)o)->%s(' % ( cl, name )
          s += first + self.genMCOArgs(args, n, len(first)) + ');\n'
          s += iden + '  return &ret;\n'        
        else :
          first = iden + '  return new %s(((%s*)o)->%s(' % ( returns, cl, name )
          s += first + self.genMCOArgs(args, n, len(first)) + '));\n'
      if ndarg : 
        if n != narg : s += '  }\n'
        else :
          if returns == 'void' : s += '  }\n  return 0;\n'
          else :                 s += '  }\n  return 0;\n'
    s += '}\n'
    return s
#----------------------------------------------------------------------------------
  def getDefaultArgs(self, args):
    n = 0
    for a in args :
      if 'default' in a : n += 1
    return n
#----------------------------------------------------------------------------------
  def genMCOArgs(self, args, narg, pad):
    s = ''
    for i in range(narg) :
      a = args[i]
      #arg = self.genArgument(a, 0);
      arg = self.genTypeName(a['type'],colon=True)
      if arg[-1] == '*' :
         if arg[-2] == ':' :   # Pointer to data member
           s += '*(%s*)arg[%d]' % (arg, i )
         else :
           s += '(%s)arg[%d]' % (arg, i )
      elif arg[-1] == ']' :
        s += '(%s)arg[%d]' % (arg, i)
      elif arg[-1] == ')' or (len(arg) > 7 and arg[-7:] == ') const'): # FIXME, the second check is a hack
        if arg.find('::*') != -1 :  # Pointer to function member
          s += '*(%s)arg[%d]' %(arg.replace('::*','::**'), i)
        elif (len(arg) > 7  and arg[-7:] == ') const') :
          s += '(%s)arg[%d]' % (arg[:-6].replace('(*)','(* const)'), i) # 2nd part of the hack
        else :
          s += '(%s)arg[%d]' % (arg, i )
      elif arg[-1] == '&' :
        s += '*(%s*)arg[%d]' % (arg[:-1], i )
      else :
        s += '*(%s*)arg[%d]' % (arg, i )
      if i != narg - 1 : s += ',\n' + pad*' '
    return s
#----------------------------------------------------------------------------------
  def genMethodDecl(self, attrs, args):
    return self.genMCODecl( 'method', '', attrs, args )
#----------------------------------------------------------------------------------
  def genMethodBuild(self, attrs, args):
    return self.genMCOBuild( 'method', attrs['name'], attrs, args )
#----------------------------------------------------------------------------------
  def genMethodDef(self, attrs, args):
    return self.genMCODef( 'method', attrs['name'], attrs, args )
#----------------------------------------------------------------------------------
  def genConstructorDecl(self, attrs, args):
    return self.genMCODecl( 'constructor', '', attrs, args )
#----------------------------------------------------------------------------------
  def genConstructorBuild(self, attrs, args):
    return self.genMCOBuild( 'constructor', attrs['name'], attrs, args )
#----------------------------------------------------------------------------------
  def genConstructorDef(self, attrs, args):
    cl  = self.genTypeName(attrs['context'], colon=True)
    clt = string.translate(str(cl), self.transtable)
    id  = attrs['id']
    if len(args) :
      s = 'static void* constructor%s( void* mem, const std::vector<void*>& arg, void*) {\n' %( id, )
    else :
      s = 'static void* constructor%s( void* mem, const std::vector<void*>&, void*) {\n' %( id, )
    if 'pseudo' in attrs :
      s += '  return ::new(mem) %s( *(__void__*)0 );\n' % ( cl )
    else :
      ndarg = self.getDefaultArgs(args)
      narg  = len(args)
      for n in range(narg-ndarg, narg+1) :
        if ndarg :
          if n == narg-ndarg :  s += '  if ( arg.size() == %d ) {\n  ' % n
          else               :  s += '  else if ( arg.size() == %d ) { \n  ' % n
        first = '  return ::new(mem) %s(' % ( cl )
        s += first + self.genMCOArgs(args, n, len(first)) + ');\n'
        if ndarg : 
          if n != narg : s += '  }\n'
          else :         s += '  }\n  return 0;\n'
    s += '}\n'
    return s
#----------------------------------------------------------------------------------
  def genDestructorDef(self, attrs, childs):
    cl = self.genTypeName(attrs['context'])
    return 'static void* destructor%s(void * o, const std::vector<void*>&, void *) {\n  ((::%s*)o)->~%s(); return 0;\n}' % ( attrs['id'], cl, attrs['name'] )
#----------------------------------------------------------------------------------
  def genDestructorBuild(self, attrs, childs):
    if self.isUnnamedType(self.xref[attrs['context']]['attrs'].get('name')) or \
       self.checkAccessibleType(self.xref[attrs['context']]) : return ''
    mod = self.genModifier(attrs,None)
    id       = attrs['id']
    s = '  .AddFunctionMember(%s, "~%s", destructor%s, 0, 0, %s | DESTRUCTOR )' % (self.genTypeID(id), attrs['name'], attrs['id'], mod)
    s += self.genCommentProperty(attrs)
    return s
#----------------------------------------------------------------------------------
  def genOperatorMethodDecl( self, attrs, args ) :
    if attrs['name'][0].isalpha() : name = 'operator '+ attrs['name']
    else                          : name = 'operator' + attrs['name'] 
    return self.genMCODecl( 'operator', name, attrs, args )    
#----------------------------------------------------------------------------------
  def genOperatorMethodBuild( self, attrs, args ) :
    if attrs['name'][0].isalpha() : name = 'operator '+ attrs['name']
    else                          : name = 'operator' + attrs['name'] 
    return self.genMCOBuild( 'operator', name, attrs, args )    
#----------------------------------------------------------------------------------
  def genOperatorMethodDef( self, attrs, args ) :
    if attrs['name'][0].isalpha() : name = 'operator '+ attrs['name']
    else                          : name = 'operator' + attrs['name']
    if name[-1] == '>' and name.find('<') != -1 : name = name[:name.find('<')]
    return self.genMCODef( 'operator', name, attrs, args )    
#----------------------------------------------------------------------------------
  def genConverterDecl( self, attrs, args ) :
    return self.genMCODecl( 'converter', 'operator '+attrs['name'], attrs, args )    
#----------------------------------------------------------------------------------
  def genConverterBuild( self, attrs, args ) :
    return self.genMCOBuild( 'converter', 'operator '+self.genTypeName(attrs['returns'],enum=True,const=True), attrs, args )    
#----------------------------------------------------------------------------------
  def genConverterDef( self, attrs, args ) :
    # If this is a conversion operator to pointer to function member we will need
    # to create a typedef for the typename which is needed in the stub function
    tdf = 'operator '+self.genTypeName(attrs['returns'])
    t1 = self.xref[attrs['returns']]
    if t1['elem'] == 'PointerType':
      t2 = self.xref[t1['attrs']['type']]
      if t2['elem'] == 'MethodType':
        tdf = self.genTypeName(attrs['returns']).replace('*)(','* TDF%s)('%attrs['id'])
    return self.genMCODef( 'converter', tdf, attrs, args )    
#----------------------------------------------------------------------------------
  def genEnumValue(self, attrs):
    return '%s = %s' % (attrs['name'], attrs['init'])
#----------------------------------------------------------------------------------
  def genBaseClassBuild(self, clf, b ):
    mod = b['access'].upper()
    if 'virtual' in b and b['virtual'] == '1' : mod = 'VIRTUAL | ' + mod
    return '  .AddBase(%s, BaseOffset< %s, %s >::Get(), %s)' %  (self.genTypeID(b['type']), clf, self.genTypeName(b['type'],colon=True), mod)
#----------------------------------------------------------------------------------
  def enhanceClass(self, attrs):
    if self.isUnnamedType(attrs['name']) or self.checkAccessibleType(self.xref[attrs['id']]) : return
    # Default constructor
    if 'members' in attrs : members = attrs['members'].split()
    else                  : members = []
    for m in members :
      if self.xref[m]['elem'] == 'Constructor' :
        args  = self.xref[m]['subelems']
        if len(args) > 0 and 'default' in args[0] :
          id = u'_x%d' % self.x_id.next()
          new_attrs = self.xref[m]['attrs'].copy()
          new_attrs['id'] = id
          new_attrs['artificial'] = 'true'
          self.xref[id] = {'elem':'Constructor', 'attrs':new_attrs,'subelems':[] }
          attrs['members'] += u' ' + id
        elif len(args) == 1 and self.genTypeName(args[0]['type']) == '__void__&' :
          id = u'_x%d' % self.x_id.next()
          new_attrs = self.xref[m]['attrs'].copy()
          new_attrs['id'] = id
          new_attrs['pseudo'] = True
          new_attrs['artificial'] = 'true'
          self.xref[id] = {'elem':'Constructor', 'attrs':new_attrs,'subelems':[] }
          attrs['members'] += u' ' + id
        elif len(args) == 0 and 'abstract' not in attrs and \
             'access' not in self.xref[m]['attrs'] and not self.isDestructorNonPublic(attrs['id']):
          # NewDel functions extra function
          id = u'_x%d' % self.x_id.next()
          new_attrs = { 'id':id, 'context':attrs['id'], 'artificial':'true' }
          self.xref[id] = {'elem':'GetNewDelFunctions', 'attrs':new_attrs,'subelems':[] }
          attrs['members'] += u' ' + id    
    # Bases extra function
    if 'bases' in attrs and attrs['bases'] != '':
      id = u'_x%d' % self.x_id.next()
      new_attrs = { 'id':id, 'context':attrs['id'], 'artificial':'true' }
      self.xref[id] = {'elem':'GetBasesTable', 'attrs':new_attrs,'subelems':[] }
      if 'members' in attrs : attrs['members'] += u' ' + id
      else                  : attrs['members'] = u' '+ id   
    # Container extra functions
    type = getContainerId( self.genTypeName(attrs['id']) )[1]
    if 'extra' in attrs and 'type' in attrs['extra'] : type = attrs['extra']['type']
    if type :
      #--The new stuff from CollectionProxy--------
      id = u'_x%d' % self.x_id.next()
      new_attrs = { 'id':id, 'context':attrs['id'], 'artificial':'true' }
      self.xref[id] = {'elem':'CreateCollFuncTable', 'attrs':new_attrs,'subelems':[] }
      if 'members' in attrs : attrs['members'] += u' ' + id
      else                  : attrs['members'] = u' ' + id
#----CollectionProxy stuff--------------------------------------------------------
  def genCreateCollFuncTableDecl( self, attrs, args ) :
    return 'static void* method%s( void*, const std::vector<void*>&, void* ); ' % (attrs['id'])
  def genCreateCollFuncTableBuild( self, attrs, args ) :
    mod = self.genModifier(attrs, None)
    return '  .AddFunctionMember<void*(void)>("createCollFuncTable", method%s, 0, 0, %s)' % ( attrs['id'], mod)
  def genCreateCollFuncTableDef( self, attrs, args ) :
    cl       = self.genTypeName(attrs['context'], colon=True)
    clt      = string.translate(str(cl), self.transtable)
    t        = getTemplateArgs(cl)[0]
    s  = 'static void* method%s( void*, const std::vector<void*>&, void*)\n{\n' %( attrs['id'], )
    s += '  return ROOT::Reflex::Proxy< %s >::Generate();\n' % (cl,)
    s += '}\n'
    return s
#----BasesMap stuff--------------------------------------------------------
  def genGetBasesTableDecl( self, attrs, args ) :
    return 'static void* method%s( void*, const std::vector<void*>&, void* ); ' % (attrs['id'])
  def genGetBasesTableBuild( self, attrs, args ) :
    mod = self.genModifier(attrs, None)
    return '  .AddFunctionMember<void*(void)>("__getBasesTable", method%s, 0, 0, %s)' % (attrs['id'], mod)
  def genGetBasesTableDef( self, attrs, args ) :
    cid      = attrs['context']
    cl       = self.genTypeName(cid, colon=True)
    clt      = string.translate(str(cl), self.transtable)
    s  = 'static void* method%s( void*, const std::vector<void*>&, void*)\n{\n' %( attrs['id'], )
    s += '  static std::vector<std::pair<ROOT::Reflex::Base, int> > s_bases;\n'
    s += '  if ( !s_bases.size() ) {\n'
    bases = []
    self.getAllBases( cid, bases ) 
    for b in bases :
      bname = self.genTypeName(b[0],colon=True)
      bname2 = self.genTypeName(b[0])
      s += '    s_bases.push_back(std::make_pair(ROOT::Reflex::Base( ROOT::Reflex::TypeBuilder("%s"), ROOT::Reflex::BaseOffset< %s,%s >::Get(),%s), %d));\n' % (bname2, cl, bname, b[1], b[2])
    s += '  }\n  return &s_bases;\n' 
    s += '}\n'
    return s
#----Constructor/Destructor stuff--------------------------------------------------------
  def checkOperators(self,cid):
    opnewc = 0
    plopnewc = 0
    opnewa = 0
    plopnewa = 0
    attrs = self.xref[cid]['attrs']
    for m in attrs.get('members').split():
      mm = self.xref[m]
      if mm['elem'] == 'OperatorMethod':
        opname = mm['attrs'].get('name')
        # we assume that 'subelems' only contains Arguments
        sems = mm['subelems']
        if opname == 'new':
          if len(sems) == 1 and self.genTypeName(sems[0]['type']) in ('size_t',): opnewc = 1
          if len(sems) == 2 and self.genTypeName(sems[0]['type']) in ('size_t',) and self.genTypeName(sems[1]['type']) in ('void*',): plopnewc = 1
        if opname == 'new []':
          if len(sems) == 1 and self.genTypeName(sems[0]['type']) in ('size_t',): opnewa = 1
          if len(sems) == 2 and self.genTypeName(sems[0]['type']) in ('size_t',) and self.genTypeName(sems[1]['type']) in ('void*',): plopnewa = 1
    newc = ''
    newa = ''
    if opnewc and not plopnewc: newc = '_np'
    elif not opnewc and plopnewc : newc = '_p'
    if opnewa and not plopnewa: newa = '_np'
    elif not opnewa and plopnewa : newa = '_p'
    return (newc, newa)
#----Constructor/Destructor stuff--------------------------------------------------------
  def genGetNewDelFunctionsDecl( self, attrs, args ) :
    return 'static void* method%s( void*, const std::vector<void*>&, void* ); ' % (attrs['id'])
  def genGetNewDelFunctionsBuild( self, attrs, args ) :
    mod = self.genModifier(attrs, None)  
    return '  .AddFunctionMember<void*(void)>("__getNewDelFunctions", method%s, 0, 0, %s)' % (attrs['id'], mod)
  def genGetNewDelFunctionsDef( self, attrs, args ) :
    cid      = attrs['context']
    cl       = self.genTypeName(cid, colon=True)
    clt      = string.translate(str(cl), self.transtable)
    (newc, newa) = self.checkOperators(cid)
    s  = 'static void* method%s( void*, const std::vector<void*>&, void*)\n{\n' %( attrs['id'] )
    s += '  static NewDelFunctions s_funcs;\n'
    s += '  s_funcs.fNew         = NewDelFunctionsT< %s >::new%s_T;\n' % (cl, newc)
    s += '  s_funcs.fNewArray    = NewDelFunctionsT< %s >::newArray%s_T;\n' % (cl, newa)
    s += '  s_funcs.fDelete      = NewDelFunctionsT< %s >::delete_T;\n' % cl
    s += '  s_funcs.fDeleteArray = NewDelFunctionsT< %s >::deleteArray_T;\n' % cl
    s += '  s_funcs.fDestructor  = NewDelFunctionsT< %s >::destruct_T;\n' % cl
    s += '  return &s_funcs;\n'
    s += '}\n'
    return s
#----------------------------------------------------------------------------------
  def getBases( self, cid ) :
    if 'bases' in self.xref[cid] :
      return self.xref[cid]['bases']
    elif 'bases' in self.xref[cid]['attrs'] :
      bases = []
      for b in self.xref[cid]['attrs']['bases'].split() :
        access = 'public'
        if b[:10] == 'protected:' : b = b[10:]; access = 'protected'
        if b[:8]  == 'private:'   : b = b[8:]; access = 'private'
        bases.append( {'type': b, 'access': access, 'virtual': '-1' } )
      return bases
    else :
      return []
#----------------------------------------------------------------------------------
  def getAllBases( self, cid, bases, level = 0, access = 'public', virtual = False ) :
    for b in self.getBases( cid ) :
      id = b['type']
      if id not in [ bid[0] for bid in bases] :
        if access == 'public' : access = b['access']
        if not virtual : virtual = ( b['virtual'] == '1' )
        mod = access.upper()
        if virtual : mod = 'VIRTUAL |' + mod
        bases.append( [id,  mod, level] )
        self.getAllBases( id, bases, level+1, access, virtual )
#----------------------------------------------------------------------------------
  def completeClass(self, attrs):
    # Complete class with "instantiated" templated methods or constructors
    if 'members' in attrs : members = attrs['members'].split()
    else                  : members = []
    cid = attrs['id']
    for c in self.classes :
      if c['context'] == cid and c['id'] not in members :
        attrs['members'] += u' ' + c['id']
    for m in self.methods :
      if m['context'] == cid and m['id'] not in members :
        # replace the mame by the complete templated name. Use the demangle module for that
        if 'mangled' in m and m['name'].isalpha() :
          mm = m['mangled'][2:]
          dname = gccdemangler.demangle_name(mm)
          dret  = gccdemangler.demangle_type(mm[dname[0]:])
          if dname[3] : mret  = mm[dname[0]:dname[0]+dret[0]]
          else        : mret  = ''
          if [mret.find(t)!= -1 for t in ['T_']+['T%d_'%i for i in range(10)]].count(True) :
            fname =  dname[1][dname[1].rfind('::' + m['name'])+2:]
            m['name'] = fname
        attrs['members'] += u' ' + m['id']
#---------------------------------------------------------------------------------------
def getContainerId(c):
  if   c[-8:] == 'iterator' : return ('NOCONTAINER','')
  if   c[:10] == 'std::deque'   :            return ('DEQUE','list')
  elif c[:9]  == 'std::list'    :            return ('LIST','list')
  elif c[:8]  == 'std::map'     :            return ('MAP','map')
  elif c[:13] == 'std::multimap':            return ('MULTIMAP','map')
  elif c[:19] == '__gnu_cxx::hash_map':      return ('HASHMAP','map')
  elif c[:24] == '__gnu_cxx::hash_multimap': return ('HASHMULTIMAP','map')
  elif c[:16] == 'stdext::hash_map':         return ('HASHMAP','map')
  elif c[:21] == 'stdext::hash_multimap':    return ('HASHMULTIMAP','map')    
  elif c[:10] == 'std::queue'   :            return ('QUEUE','queue')
  elif c[:8]  == 'std::set'     :            return ('SET','set')
  elif c[:13] == 'std::multiset':            return ('MULTISET','set')
  elif c[:19] == '__gnu_cxx::hash_set':      return ('HASHSET','set')
  elif c[:24] == '__gnu_cxx::hash_multiset': return ('HASHMULTISET','set')
  elif c[:16] == 'stdext::hash_set':         return ('HASHSET','set')
  elif c[:21] == 'stdext::hash_multiset':    return ('HASHMULTISET','set')
  elif c[:10] == 'std::stack'   :            return ('STACK','stack')
  elif c[:11] == 'std::vector'  :            return ('VECTOR','vector')
  else : return ('NOCONTAINER','')
#---------------------------------------------------------------------------------------
stldeftab = {}
stldeftab['deque']        = '=','std::allocator'
stldeftab['list']         = '=','std::allocator'
stldeftab['map']          = '=','=','std::less','std::allocator'
stldeftab['multimap']     = '=','=','std::less','std::allocator'
stldeftab['queue']        = '=','std::deque'
stldeftab['set']          = '=','std::less','std::allocator'
stldeftab['multiset']     = '=','std::less','std::allocator'
stldeftab['stack']        = '=','std::deque'
stldeftab['vector']       = '=','std::allocator'
stldeftab['basic_string'] = '=','std::char_traits','std::allocator'
#stldeftab['basic_ostream']= '=','std::char_traits'
#stldeftab['basic_istream']= '=','std::char_traits'
#stldeftab['basic_streambuf']= '=','std::char_traits'
if sys.platform == 'win32' :
  stldeftab['hash_set']      = '=', 'stdext::hash_compare', 'std::allocator'
  stldeftab['hash_multiset'] = '=', 'stdext::hash_compare', 'std::allocator'
  stldeftab['hash_map']      = '=', '=', 'stdext::hash_compare', 'std::allocator'
  stldeftab['hash_multimap'] = '=', '=', 'stdext::hash_compare', 'std::allocator'
else :
  stldeftab['hash_set']      = '=','__gnu_cxx::hash','std::equal_to','std::allocator'
  stldeftab['hash_multiset'] = '=','__gnu_cxx::hash','std::equal_to','std::allocator'
  stldeftab['hash_map']      = '=','=','__gnu_cxx::hash','std::equal_to','std::allocator'
  stldeftab['hash_multimap'] = '=','=','__gnu_cxx::hash','std::equal_to','std::allocator'  
#---------------------------------------------------------------------------------------
def getTemplateArgs( cl ) :
  if cl.find('<') == -1 : return []
  args, cnt = [], 0
  for s in string.split(cl[cl.find('<')+1:cl.rfind('>')],',') :
    if   cnt == 0 : args.append(s)
    else          : args[-1] += ','+ s
    cnt += s.count('<')+s.count('(')-s.count('>')-s.count(')')
  if args[-1][-1] == ' ' : args[-1] = args[-1][:-1]
  return args
#---------------------------------------------------------------------------------------
def getTemplateArgString( cl ) :
  bc = 0
  if cl[-1] != '>' : return ''
  for i in range( len(cl)-1, -1, -1) :
    if   cl[i] == '>' : bc += 1
    elif cl[i] == '<' : bc -= 1
    if bc == 0 : return cl[i:]
  return ''
#---------------------------------------------------------------------------------------
def normalizeClassAllTempl(name)   : return normalizeClass(name,True)
def normalizeClassNoDefTempl(name) : return normalizeClass(name,False)
def normalizeClass(name,alltempl) :
  names, cnt = [], 0
  for s in string.split(name,'::') :
    if cnt == 0 : names.append(s)
    else        : names[-1] += '::' + s
    cnt += s.count('<')-s.count('>')
  if alltempl : return string.join(map(normalizeFragmentAllTempl,names),'::')
  else        : return string.join(map(normalizeFragmentNoDefTempl,names),'::')
#--------------------------------------------------------------------------------------
def normalizeFragmentAllTempl(name)   : return normalizeFragment(name,True)
def normalizeFragmentNoDefTempl(name) : return normalizeFragment(name) 
def normalizeFragment(name,alltempl=False) :
  name = name.strip()
  if name.find('<') == -1  : 
    nor =  name
    for e in [ ['long long unsigned int', 'unsigned long long'],
             ['long long int',          'long long'],
             ['unsigned short int',     'unsigned short'],
             ['short unsigned int',     'unsigned short'],
             ['short int',              'short'],
             ['long unsigned int',      'unsigned long'],
             ['unsigned long int',      'unsigned long'],
             ['long int',               'long']] :
      nor = nor.replace(e[0], e[1])
    return nor
  else                     : clname = name[:name.find('<')]
  if name.rfind('>') == -1 : suffix = ''
  else                     : suffix = name[name.rfind('>')+1:]
  args = getTemplateArgs(name)
  if alltempl :
    nor = clname + '<' + string.join(map(normalizeClassAllTempl,args),',')
  else :     
    if clname in stldeftab :
      # select only the template parameters different from default ones
      sargs = []
      for i in range(len(args)) :  
        if args[i].find(stldeftab[clname][i]) == -1 : sargs.append(args[i])
      nor = clname + '<' + string.join(map(normalizeClassNoDefTempl,sargs),',')
    else :
      nor = clname + '<' + string.join(map(normalizeClassNoDefTempl,args),',')
  if nor[-1] == '>' : nor += ' >' + suffix
  else              : nor += '>' + suffix
  return nor
#--------------------------------------------------------------------------------------
def clean(a) :
  r = []
  for i in a :
	if i not in r : r.append(i)
  return r
