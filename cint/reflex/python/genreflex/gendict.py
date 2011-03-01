# Copyright CERN, CH-1211 Geneva 23, 2004-2010, All rights reserved.
#
# Permission to use, copy, modify, and distribute this software for any
# purpose is hereby granted without fee, provided that this copyright and
# permissions notice appear in all copies and derivatives.
#
# This software is provided "as is" without express or implied warranty.

import xml.parsers.expat
import os, sys, string, time, re
import gccdemangler

class genDictionary(object) :
#----------------------------------------------------------------------------------
  def __init__(self, hfile, opts, gccxmlvers):
    self.classes    = []
    self.namespaces = []
    self.typeids    = []
    self.fktypeids  = []
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
    self.interpreter= opts.get('interpreter',False)
    self.quiet      = opts.get('quiet',False)
    self.resolvettd = opts.get('resolvettd',True)
    self.xref       = {}
    self.xrefinv    = {}
    self.cppClassSelect    = {}
    self.cppVariableSelect = {}
    self.cppEnumSelect     = {}
    self.cppFunctionSelect = {}
    self.last_id    = ''
    self.transtable = string.maketrans('<>&*,: ().$-[]', '__rp__s___dm__')
    self.ignoremeth = ('rbegin', 'rend', '_Eq','_Lt', 'value_comp')
    self.x_id       = iter(xrange(sys.maxint))
    self.errors     = 0
    self.warnings   = 0
    self.comments           = opts.get('comments', False)
    self.iocomments     = opts.get('iocomments', False)
    self.no_membertypedefs  = opts.get('no_membertypedefs', False)
    self.generated_shadow_classes = []
    self.selectionname      = 'Reflex::Selection'
    self.unnamedNamespaces = []
    self.globalNamespaceID = ''
    self.typedefs_for_usr = []
    self.gccxmlvers = gccxmlvers
    self.split = opts.get('split', '')
    # The next is to avoid a known problem with gccxml that it generates a
    # references to id equal '_0' which is not defined anywhere
    self.xref['_0'] = {'elem':'Unknown', 'attrs':{'id':'_0','name':''}, 'subelems':[]}
    self.TObject_id = ''
#----------------------------------------------------------------------------------
  def addTemplateToName(self, attrs):
    if attrs['name'].find('>') == -1 and 'demangled' in attrs :
      # check whether this method is templated; GCCXML will
      # not pass the real name foo<int> but only foo"
      demangled = attrs['demangled']
      posargs = demangled.rfind('(')
      if posargs and posargs > 1 \
             and demangled[posargs - 1] == '>' \
             and (demangled[posargs - 2].isalnum() \
                  or demangled[posargs - 2] == '_') :
        posname = demangled.find(attrs['name'] + '<');
        if posname :
          reui = re.compile('\\b(unsigned)(\\s+)?([^\w\s])')
          name1 = demangled[posname : posargs]
          attrs['name'] = reui.sub('unsigned int\\3', name1)
#----------------------------------------------------------------------------------
  def patchTemplateName(self, attrs, elem):
    if 'name' not in attrs: return
    name = attrs['name']
    postmpltend = name.rfind('>')
    if postmpltend != -1 :
      # check whether this entity is templated and extract the template parameters
      postmpltend = len(name)
      if elem in ('Function','OperatorFunction','Constructor','Method','OperatorMethod'):
        postmpltend = name.rfind('(')
      postmplt = -1
      if postmpltend and postmpltend > 1 \
               and name[postmpltend - 1] == '>':
        postmplt = name.find('<')
      if postmplt != -1:
        postmplt += 1
        postmpltend -= 1
        # replace template argument "12u" or "12ul" by "12":
        rep = re.sub(r"\b([\d]+)ul?\b", '\\1', name[postmplt:postmpltend])
        # replace -0x00000000000000001 by -1
        rep = re.sub(r"-0x0*([1-9A-Fa-f][0-9A-Fa-f]*)\b", '-\\1', rep)
        name = name[:postmplt] + rep + name[postmpltend:]
        attrs['name'] = name
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
      self.patchTemplateName(attrs, name)
      self.classes.append(attrs)
      if 'name' in attrs and attrs['name'] == 'TObject' :
        self.TObject_id = attrs['id']
    elif name in ('Function',) :
      self.addTemplateToName(attrs)
      self.patchTemplateName(attrs, name)
      self.functions.append(attrs)
    elif name in ('Enumeration',) :
      self.enums.append(attrs)
    elif name in ('Variable',) :
      self.variables.append(attrs)
    elif name in ('OperatorFunction',) :
      if 'name' in attrs:
        if attrs['name'][0:8] == 'operator':
          if attrs['name'][8] == ' ':
            if not attrs['name'][9].isalpha() :
              attrs['name']= 'operator' + attrs['name'][9:]
        else :
          if attrs['name'][0].isalpha(): attrs['name'] = 'operator ' + attrs['name']
          else                         : attrs['name'] = 'operator' + attrs['name']
      self.patchTemplateName(attrs, name)
      attrs['operator'] = 'true'
      self.addTemplateToName(attrs)
      self.functions.append(attrs)
    elif name in ('Constructor','Method','OperatorMethod') :
      if 'name' in attrs and attrs['name'][0:3] != '_ZT' :
        self.addTemplateToName(attrs)
        self.patchTemplateName(attrs, name)
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
  def findSpecialNamespace(self):
    for ns in self.namespaces:
      if 'name' not in ns or ns['name'].find('.') != -1:
        self.unnamedNamespaces.append(ns['id'])
      elif ns['name'] == '::' :
        self.globalNamespaceID = ns['id']
#----------------------------------------------------------------------------------
# This function is not used anymore, because it had a problem with templated types
# showing up as members of a scope in the gccxml generated output. If gccxml puts
# templated types as members this function should be resurrected for performance
# reasons. It shall be called from self.parse instead of 'for col in [self.class...'
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
  def inScope(self, item, ctxt):
    ictxt = item.get('context')
    if ictxt :
      if ictxt == ctxt : return 1
      else             : return self.inScope(self.xref[ictxt]['attrs'], ctxt)
    return 0
#----------------------------------------------------------------------------------
  def addCppClassSelect(self, key, value ) : self.cppClassSelect[key] = value
  def addCppStructSelect(self, key, value ) : self.cppClassSelect[key] = value
  def addCppFunctionSelect(self, key, value ) : self.cppFunctionSelect[key] = value
  def addCppVariableSelect(self, key, value ) : self.cppVariableSelect[key] = value
  def addCppEnumerationSelect(self, key, value ) : self.cppEnumSelect[key] = value  
#----------------------------------------------------------------------------------
  def parse(self, file) :
    p = xml.parsers.expat.ParserCreate()
    p.StartElementHandler = self.start_element
    f = open(file)
    p.ParseFile(f)
    f.close()
    cppselid = '';
    cppsellen = len(self.selectionname)+2
    # get the id of the selection namespace
    for n in self.namespaces: 
      if self.genTypeName(n['id']) == self.selectionname :
        cppselid = n.get('id')
        break
    # for all classes, variables etc. check if they are in the seleciton ns and add them if
    for col in [self.classes, self.variables, self.enums, self.functions ]:
      for it in col:
        if self.inScope(it, cppselid) :
          cid = it['id']
          funname = 'addCpp'+self.xref[cid]['elem']+'Select'
          if funname in dir(self):
            self.__class__.__dict__[funname](self, self.genTypeName(cid, _useCache=False)[cppsellen:], cid)
    self.tryCppSelections()
    self.findSpecialNamespace()
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
  def resolveTypedefName( self, name ) :
    notname = ['*','<','>',',',':',' ']
    for td in self.typedefs :
      tdname = self.genTypeName(td['id'])
      f = name.find(tdname)
      if f != -1 :
        g = f + len(tdname)
        if (f == 0 or name[f-1] in notname) and (g == len(tdname) or name[g] in notname) :
          defname = self.genTypeName(td['type'])
          if defname[len(defname)-1] == '>' : defname += ' '
          name = self.resolveTypedefName(name.replace(tdname, defname, 1))
    return name
#----------------------------------------------------------------------------------
  def genFakeTypedef( self, cid, name ) :
    nid = cid+'f'
    while nid in self.fktypeids : nid += 'f'
    catt = self.xref[cid]['attrs']
    attrs = {'name':name,'id':nid,'type':catt['id'],'context':self.globalNamespaceID}
    for i in ['location','file','line'] : attrs[i] = catt[i]
    self.typedefs.append(attrs)
    self.xref[attrs['id']] = {'elem':'Typedef', 'attrs':attrs, 'subelems':[]}
    self.fktypeids.append(nid)
#----------------------------------------------------------------------------------
  def resolveSelectorTypedefs( self, sltor ):
    newselector = []
    for sel in sltor :
      if not sel.has_key('used'):
        attrs = sel['attrs']
        for n in ['name', 'pattern']:
          if attrs.has_key(n) and attrs[n].find('<') != -1 :
            newname = self.resolveTypedefName(attrs[n])
            if newname != attrs[n]:
              sel['attrs']['o_'+n] = attrs[n]
              sel['attrs'][n] = newname
              newselector.append(sel)
    return newselector
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

        # Filter any non-public data members for minimal interpreter dict
        if self.interpreter:
          cxref = self.xref[c['id']]
          # assumes that the default is "public"
          if cxref.has_key('attrs') and 'access' in cxref['attrs'] :
            continue

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
            if catt not in selec :
              t['fullname'] = self.genTypeName(t['id'])
              if not self.quiet:
                print '--->> genreflex: INFO: Using typedef %s to select class %s' % (t['fullname'], self.genTypeName(catt['id']))
              selec.append(catt)
              self.typedefs_for_usr.append(t)
              if match[0].has_key('fields') :
                # copy all fields selection attrs over to the underlying class, see sav #49472
                # this part is needed for selfields which checks the selection, not the class's attrs
                clsname = self.genTypeName(catt['id'])
                newselattrs = {'name': clsname, 'n_name': self.selector.genNName(clsname)}
                self.selector.sel_classes.append({'fields': match[0]['fields'], 'attrs': newselattrs, 'used': 1, 'methods':[]})
            elif match[0].has_key('fields') :
              # copy all fields selection attrs over to the underlying class, see sav #49472
              clname = self.genTypeName(catt['id'])
              for c in self.selector.sel_classes :
                attrs = c['attrs']
                if 'n_name' in attrs and attrs['n_name'] == clname \
                      or 'n_pattern' in attrs and self.selector.matchpattern(clname, attrs['n_pattern']) :
                  c['fields'] += match[0]['fields']
      if self.resolvettd :
        newselector = self.resolveSelectorTypedefs( self.selector.sel_classes )
        if newselector:
          for c in self.classes:
            match = self.selector.matchclassTD( self.genTypeName(c['id']), self.files[c['file']]['name'], newselector )
            if match[0] and not match[1] :
              n = 'name'
              if 'pattern' in match[0] : n = 'pattern'
              if not self.quiet:
                print '--->> genreflex: INFO: Replacing selection %s "%s" with "%s"' % ( n, match[0]['o_'+n], match[0][n] )
              c['extra'] = match[0]
              if c not in selec : selec.append(c)
              if n == 'name' : self.genFakeTypedef(c['id'], match[0]['o_name'])
      # Filter STL implementation specific classes
      selec =  filter( lambda c: c.has_key('name'), selec)  # unamed structs and unions
      # Filter internal GCC classes
      selec =  filter( lambda c: c['name'].find('_type_info_pseudo') == -1, selec)
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
    classes =  filter( lambda c: self.genTypeName(c['id'])[:6] != 'std::_', classes)
    classes =  filter( lambda c: c.has_key('name'), classes)  # unamed structs and unions
    # Filter internal GCC classes
    classes =  filter( lambda c: c['name'].find('_type_info_pseudo') == -1, classes)
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
        id = f['id']
        funcname = self.genTypeName(id)
        attrs = self.xref[id]['attrs']
        demangled = attrs.get('demangled')
        returns = ''
        if 'returns' in attrs: returns = self.genTypeName(attrs['returns'])
        lenreturns = len(returns)
        if demangled and len(demangled) :
          if lenreturns and demangled[0:lenreturns] == returns:
            demangled = demangled[lenreturns:]
            while demangled[0] == ' ': demangled = demangled[1:]
        else :
          demangled = ""
        if self.selector.selfunction( funcname, demangled ) and not self.selector.excfunction( funcname, demangled ) :
          selec.append(f)
        elif 'extra' in f and f['extra'].get('autoselect') and f not in selec:
          selec.append(f)
    return selec
#----------------------------------------------------------------------------------
  def selenums(self, sel) :
    selec = []
    self.selector = sel  # remember the selector
    if self.selector :
      for e in self.enums :
        # Filter any non-public data members for minimal interpreter dict
        if self.interpreter:
          exref = self.xref[e['id']]
          # assumes that the default is "public"
          if exref.has_key('attrs') and 'access' in exref['attrs'] :
            continue

        ename = self.genTypeName(e['id'])
        if self.selector.selenum( ename ) and not self.selector.excenum( ename ) :
          selec.append(e)
        elif 'extra' in e and e['extra'].get('autoselect') and e not in selec:
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
        elif 'extra' in v and v['extra'].get('autoselect') and v not in selec:
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
            xref = self.xref[m]
            if xref['elem'] in ['Field','Typedef'] and xref['attrs']['access']=="public":
              type = xref['attrs']['type']
              self.getdependent(type, types)
	    elif xref['elem'] in ['Method','OperatorMethod','Constructor'] \
                     and self.isMethodReallyPublic(m):
              if 'returns' in xref['attrs']:
                type = xref['attrs']['returns']
                self.getdependent(type, types)
	      for arg in  xref['subelems']:
		  type = arg['type']
		  self.getdependent(type, types)
	    else:
	      pass #print "Doing nothing for element:", self.xref[m]['elem']
        if 'bases' in attrs :
          for b in attrs['bases'].split() :
            if b[:10] == 'protected:' : b = b[10:]
            if b[:8]  == 'private:'   : b = b[8:]
            self.getdependent(b, types)
#----------------------------------------------------------------------------------
  def sortselclasses(self, l):
    nolit = [' ', ':', '<', '>']
    l2 = []
    for x in l:
      ap = 1
      for i in range(len(l2)):
        l2ifn = l2[i]['fullname']
        xfn = x['fullname']
        bpos = l2ifn.find(xfn)
        epos = bpos + len(xfn)
        if bpos != -1 and ( bpos == 0 or l2ifn[bpos-1] in nolit  ) and ( epos == len(l2ifn) or l2ifn[epos] in nolit ) :
          l2.insert(i,x)
          ap = 0
          break
      if ap : l2.append(x)
    return l2
#----------------------------------------------------------------------------------
  def generate(self, file, selclasses, selfunctions, selenums, selvariables, cppinfo, ioReadRules = None, ioReadRawRules = None) :
    for c in selclasses :  c['fullname'] = self.genTypeName(c['id'])
    selclasses = self.sortselclasses(selclasses)
    names = []
    f = open(file,'w') 
    f.write(self.genHeaders(cppinfo))

    #------------------------------------------------------------------------------
    # Process includes relevent to the IO rules
    #------------------------------------------------------------------------------
    if ioReadRules or ioReadRawRules:
      f.write( '#include "TBuffer.h"\n' )
      f.write( '#include "TVirtualObject.h"\n' )
      f.write( '#include <vector>\n' )
      f.write( '#include "TSchemaHelper.h"\n\n' )

      includes = self.getIncludes( ioReadRules, ioReadRawRules )
      for inc in includes:
        f.write( '#include <%s>\n' % (inc,) )
      f.write( '\n' )

  
    #------------------------------------------------------------------------------
    # Process ClassDef implementation before writing: sets 'extra' properties
    #------------------------------------------------------------------------------
    classDefImpl = ClassDefImplementation(selclasses, self)

    #------------------------------------------------------------------------------
    # Process Class_Version implementation before writing: sets 'extra' properties
    #------------------------------------------------------------------------------
    Class_VersionImplementation(selclasses,self)

    f_buffer = ''
    # Need to specialize templated class's functions (e.g. A<T>::Class())
    # before first instantiation (stubs), so classDefImpl before stubs.
    if self.split.find('classdef') >= 0:
      posExt = file.rfind('.')
      if posExt > 0:
        cdFileName = file[0:posExt] + '_classdef' + file[posExt:]
      else:
        cdFileName = file + '_classdef.cpp'
      cdFile = open(cdFileName, 'w')
      cdFile.write(self.genHeaders(cppinfo))
      cdFile.write('\n')
      cdFile.write('namespace {' )
      cdFile.write(classDefImpl)
      cdFile.write('} // unnamed namespace\n')
    else :
      f_buffer += classDefImpl

    f_shadow =  '\n#ifndef __CINT__\n'
    f_shadow +=  '\n// Shadow classes to obtain the data member offsets \n'
    f_shadow += 'namespace __shadow__ {\n'
    for c in selclasses :
      if 'incomplete' not in c :
        className = c['fullname']
        if not self.quiet : print  'class '+ className

        #--------------------------------------------------------------------------
        # Get the right io rules
        #--------------------------------------------------------------------------
        clReadRules = None
        if ioReadRules and ioReadRules.has_key( className ):
          clReadRules = ioReadRules[className]
        clReadRawRules = None
        if ioReadRawRules and ioReadRawRules.has_key( className ):
          clReadRawRules = ioReadRawRules[className]

        names.append(className)
        self.completeClass( c )
        self.enhanceClass( c )
        scons, stubs   = self.genClassDict( c, clReadRules, clReadRawRules )
        f_buffer += stubs
        f_buffer += scons
        f_shadow += self.genClassShadow(c)
    f_shadow += '}\n\n'
    f_shadow +=  '\n#endif // __CINT__\n'
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
    klist= []
    m = self.xref[selection['id']]
    name = m['attrs']['name']
    parent_args = getTemplateArgs (name)
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
              if tlist and tlist[-1] != '=':
                break
              tlist.append ('=')
              klist.append (normalizeClassAllTempl (parent_args[i-1]))
            else:
              tlist.append (self.genTypeName (arg, alltempl=True))
            i += 1
    if tlist:
      name = self.xref[c['id']]['attrs']['name']
      i = name.find ('<')
      if i>=0 : name = name[:i]
      stldeftab.setdefault(name, {})[tuple (klist)] = tuple (tlist)
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
    #----Filter any method and operator for minimal POOL dict -----
    if self.pool :
      if elem in ('OperatorMethod','Converter') : return 0
      elif elem in ('Method',) :
        if attrs['name'] not in ('at','size','clear','resize') : return 0
      elif elem in ('Constructor',) :
        if len(args) > 1 : return 0
        elif len(args) == 1 :
          if self.genTypeName(args[0]['type']) != 'const '+self.genTypeName(attrs['context'])+'&' : return 0
    #----Filter any non-public data members for minimal interpreter dict -----
    if self.interpreter and elem in ('Field') and 'access' in attrs : # assumes that the default is "public"
      return 0
    #----Filter any non public method
    if attrs.get('access') in ('protected', 'private') : 
      if elem in ('Constructor','Destructor','Method','OperatorMethod','Converter') : return 0
    #----Filter any copy constructor with a private copy constructor in any base
    if elem == 'Constructor' and len(args) == 1 and 'name' in args[0] and args[0]['name'] == '_ctor_arg' :
      if self.isConstructorPrivate(attrs['context']) : return 0
    #----Filter any constructor for pure abstract classes
    if 'context' in attrs :
      if 'abstract' in self.xref[attrs['context']]['attrs'] : 
        if elem in ('Constructor',) : return 0

    if elem in ['Method', 'Constructor', 'OperatorMethod']:
      if self.hasNonPublicArgs(args):
        print "censoring method:",attrs['name']
        return 0
    #----Filter using the exclusion list in the selection file
    if self.selector and 'name' in attrs and  elem in ('Constructor','Destructor','Method','OperatorMethod','Converter') :
      context = self.genTypeName(attrs['context'])
      demangledMethod = attrs.get('demangled')
      if demangledMethod: demangledMethod = demangledMethod[len(context) + 2:]
      if self.selector.excmethod(self.genTypeName(attrs['context']), attrs['name'], demangledMethod ) : return 0
    return 1
#----------------------------------------------------------------------------------
  def isMethodReallyPublic(self,id):
    """isMethodReallyPublic checks the accessibility of the method as well as the accessibility of the types
    of arguments and return value. This is needed because C++ allows methods in a public section to be defined
    from types defined in a private/protected section.
    """
    xref = self.xref[id]
    attrs = xref['attrs']
    return (attrs['access'] == "public"
            and
            (not self.hasNonPublicArgs(xref['subelems'])) 
            and
            (not 'returns' in attrs or self.isTypePublic(attrs['returns'])))
#----------------------------------------------------------------------------------
  def hasNonPublicArgs(self,args):
    """hasNonPublicArgs will process a list of method arguments to check that all the referenced arguments in there are publically available (i.e not defined using protected or private types)."""
    for arg in args:
      type = arg["type"]
      public = self.isTypePublic(type)
      if public == 0:
        return 1
    return 0
#----------------------------------------------------------------------------------
  def isTypePublic(self, id):
    type_dict = self.xref[id]

    if type_dict['elem'] in ['PointerType','Typedef', 'ReferenceType', 'CvQualifiedType']:
      return self.isTypePublic(type_dict['attrs']['type'])
    elif type_dict['elem'] in ['FundamentalType']:
      return 1
    elif type_dict['elem'] in ['Class','Struct']:
      access=type_dict['attrs'].get('access')
      if access and access != 'public':
        return 0
      else:
        return 1
    else:
      return 1
      #raise "Unknown type category in isTypePublic",type_dict['elem']
#----------------------------------------------------------------------------------
  def tmplclasses(self, local):
    import re
    result = []
    lc_patterns = map(lambda lc: re.compile("\\b%s\\b" % lc['name']) ,
                      filter(lambda l: 'name' in l, local))
    for c in self.classes :
      if not 'name' in c: continue
      name = c['name']
      if name.find('<') == -1 : continue
      temp = name[name.find('<')+1:name.rfind('>')]
      for lc_pattern in lc_patterns :
        if lc_pattern.match(temp) : result.append(c)
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
    c += '#pragma warning ( disable : 4345 )\n'
    c += '#elif defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3)) && !defined(__INTEL_COMPILER) \n'
    c += '# pragma GCC diagnostic ignored "-Warray-bounds"\n'
    c += '#endif\n'
    c += '#include "%s"\n' % self.hfile
    c += '#ifdef CONST\n'
    c += '# undef CONST\n'
    c += '#endif\n'
    c += '#include "Reflex/Builder/ReflexBuilder.h"\n'
    c += '#include <typeinfo>\n'
    c += '\n'
    return c
#----------------------------------------------------------------------------------
  def genInstantiateDict( self, selclasses, selfunctions, selenums, selvariables) :
    c = 'namespace {\n  struct Dictionaries {\n    Dictionaries() {\n'
    c += '      Reflex::Instance initialize_reflex;\n'
    for attrs in selclasses :
      if 'incomplete' not in attrs : 
        clf = '::'+ attrs['fullname']
        clt = string.translate(str(clf), self.transtable)
        c += '      %s_dict(); \n' % (clt)
    c += self.genFunctions(selfunctions)
    c += self.genEnums(selenums)
    c += self.genVariables(selvariables)
    c += '    }\n    ~Dictionaries() {\n'
    for attrs in selclasses :
      if 'incomplete' not in attrs : 
        c += '      %s.Unload(); // class %s \n' % (self.genTypeID(attrs['id']), attrs['fullname'])
    c += '    }\n  };\n'
    c += '  static Dictionaries instance;\n}\n'
    return c
#---------------------------------------------------------------------------------
  def processIoRules( self, rules, listname ):
    sc = ''
    i = 0
    for rule in rules:
      attrs = rule['attrs']
      sc += '  rule = &%s[%d];\n' %(listname, i)
      i += 1

      sc += '  rule->fSourceClass = "%s";\n' % (attrs['sourceClass'],)

      if attrs.has_key( 'target' ):
        sc += '  rule->fTarget      = "%s";\n' % (attrs['target'],)

      if attrs.has_key( 'source' ):
        sc += '  rule->fSource      = "%s";\n' % (attrs['source'],)
        
      if rule.has_key( 'funcname' ):
        sc += '  rule->fFunctionPtr = (void *)%s;\n' % (rule['funcname'],)
        sc += '  rule->fCode        = "%s";\n' % (rule['code'].replace( '\n', '\\n' ), )

      if attrs.has_key( 'version' ):
        sc += '  rule->fVersion     = "%s";\n' % (attrs['version'],)

      if attrs.has_key( 'checksum' ):
        sc += '  rule->fChecksum    = "%s";\n' % (attrs['checksum'],)

      if attrs.has_key( 'embed' ):
        sc += '  rule->fEmbed       = %s;\n' % (attrs['embed'],)

      if attrs.has_key( 'include' ):
        sc += '  rule->fInclude     = "%s";\n' % (attrs['include'],)

      if attrs.has_key( 'attributes' ):
        sc += '  rule->fAttributes     = "%s";\n' % (attrs['attributes'],)

    return sc
#---------------------------------------------------------------------------------
  def processIOAutoVariables( self, className, mappedName, source, target, memTypes ):
    sc = '  //--- Variables added by the code generator ---\n'

    #-----------------------------------------------------------------------------
    # Write the source member ids and check if we should write the on-disk struct
    #-----------------------------------------------------------------------------
    generateOnFile = False
    sc += '#if 0\n'
    for member in source:
      sc += '  static int id_%s = oldObj->GetId("%s");\n' % (member[1], member[1])
      if member[0] != '':
        generateOnFile = True
    sc += '#endif\n'

    #-----------------------------------------------------------------------------
    # Generate the onfile structure if needed
    #-----------------------------------------------------------------------------
    if generateOnFile:
      onfileStructName = mappedName + '_Onfile'
      sc += '  struct ' + onfileStructName + ' {\n'

      #---------------------------------------------------------------------------
      # Generate the member list
      #---------------------------------------------------------------------------
      for member in source:
        if member[0] == '': continue
        sc += '    ' + member[0] + ' &' + member[1] + ';\n'

      #---------------------------------------------------------------------------
      # Generate the constructor
      #---------------------------------------------------------------------------
      sc += '    ' + onfileStructName + '( '
      start = True
      for member in source:
        if member[0] == '': continue

        if not start: sc += ', ';
        else: start = False

        sc += member[0] + ' &onfile_' + member[1]
      sc += ' ): '

      #---------------------------------------------------------------------------
      # Generate the initializer list
      #---------------------------------------------------------------------------
      start = True
      for member in source:
        if member[0] == '': continue

        if not start: sc += ', ';
        else: start = False

        sc += member[1] + '(onfile_' + member[1] + ')'
      sc += '{}\n'
      sc += '  };\n'

      #---------------------------------------------------------------------------
      # Initialize the structure - to  be changed later
      #---------------------------------------------------------------------------
      for member in source:
        sc += '  static Long_t offset_Onfile_' + mappedName
        sc += '_' + member[1] + ' = oldObj->GetClass()->GetDataMemberOffset("'
        sc += member[1] +'");\n';

      sc += '  char *onfile_add = (char*)oldObj->GetObject();\n'
      sc += '  ' + mappedName + '_Onfile onfile(\n'

      start = True
      for member in source:
        if member[0] == '': continue;

        if not start: sc += ",\n"
        else: start = False

        sc += '         '
        sc += '*(' + member[0] + '*)(onfile_add+offset_Onfile_'
        sc += mappedName + '_' + member[1] + ')'  

      sc += ' );\n\n'

    #-----------------------------------------------------------------------------
    # Write the target members
    #-----------------------------------------------------------------------------
    for member in target:
      sc += '  %s &%s = *(%s*)(target + OffsetOf(__shadow__::%s, %s));\n' % (memTypes[member], member, memTypes[member], mappedName, member)
    return sc + '\n'
#---------------------------------------------------------------------------------
  def processIoReadFunctions( self, cl, clt, rules, memTypes ):
    i = 0;
    sc = ''
    for rule in rules:
      if rule.has_key( 'code' ) and rule['code'].strip('\n').strip(' ') != '':
        funcname = 'read_%s_%d' % (clt, i)

        #--------------------------------------------------------------------------
        # Process the data members
        #--------------------------------------------------------------------------
        sourceMembers = [member.strip() for member in rule['attrs']['source'].split(';')]
        sourceMembersSpl = []
        for member in sourceMembers:
          type = ''
          elem = ''
          spl = member.split( ' ' )

          if len(spl) == 1:
            elem = member
          else:
            type = ' '.join(spl[0:len(spl)-1])
            elem = spl[len(spl)-1]
          sourceMembersSpl.append( (type, elem) )

        targetMembers = [member.strip() for member in rule['attrs']['target'].split(';')]

        #--------------------------------------------------------------------------
        # Print things out
        #--------------------------------------------------------------------------
        sc += 'void %s( char *target, TVirtualObject *oldObj )\n' % (funcname,)
        sc += '{\n'
        sc += self.processIOAutoVariables( cl, clt, sourceMembersSpl, targetMembers, memTypes )
        sc += '  %s* newObj = (%s*)target;\n' % (cl, cl)
        sc += '  //--- User\'s code ---\n'
        sc += rule['code'].strip('\n')
        sc += '\n}\n\n'
        rule['funcname'] = funcname
        i += 1
    return sc
#---------------------------------------------------------------------------------
  def processIoReadRawFunctions( self, cl, clt, rules, memTypes ):
    i = 0;
    sc = ''
    for rule in rules:
      if rule.has_key( 'code' ):
        funcname = 'readraw_%s_%d' % (clt, i)
        targetMembers = [member.strip() for member in rule['attrs']['target'].split(';')]
        sc += 'static void %s( char *target, TBuffer *oldObj )\n' % (funcname,)
        sc += '{\n'
        sc += '#if 0\n';
        sc += self.processIOAutoVariables( cl, clt, [], targetMembers, memTypes )
        sc += '  %s* newObj = (%s*)target;\n' % (cl, cl)
        sc += '  //--- User\'s code ---\n'
        sc += rule['code'].strip('\n')
        sc += '\n#endif\n'
        sc += '}\n\n'
        rule['funcname'] = funcname
        i += 1
    return sc

#---------------------------------------------------------------------------------
  def createTypeMap( self, memIds ):
    toRet = {}
    for memId in memIds:
      if self.xref[memId]['elem'] != 'Field': continue
      attrs = self.xref[memId]['attrs']
      toRet[attrs['name']] = self.genTypeName( attrs['type'] )
    return toRet
#---------------------------------------------------------------------------------
  def removeBrokenIoRules( self, cl, rules, members ):
    for rule in rules:
      if rule['attrs'].has_key( 'target' ):
        targets = [target.strip() for target in rule['attrs']['target'].split(';')]
        ok = True
        for t in targets:
          if not members.has_key( t ): ok = False
        if not ok:
          print '--->> genreflex: WARNING: IO rule for class', cl,
          print '- data member', t, 'appears on the target list but does not seem',
          print 'to be present in the target class'
          rules.remove( rule )
#---------------------------------------------------------------------------------
  def getIncludes( self, readRules, readRawRules ):
    testDict = {}
    rulesets = []

    if readRules: rulesets.append( readRules )
    if readRawRules: rulesets.append( readRawRules )

    for ruleset in rulesets:
      for ruleList in ruleset.values():
        for rule in ruleList:
          if not rule['attrs'].has_key( 'include' ):
            continue
          lst = [r.strip() for r in rule['attrs']['include'].split( ';' )]
          for r in lst:
            testDict[r] = 1
    return testDict.keys()
#---------------------------------------------------------------------------------
  def translate_typedef (self, id):
    while self.xref[id]['elem'] in ['CvQualifiedType', 'Typedef']:
      id = self.xref[id]['attrs']['type']
    return self.genTypeName(id,enum=True, const=True)
#---------------------------------------------------------------------------------
  def genClassDict(self, attrs, ioReadRules, ioReadRawRules):
    members, bases = [], []
    cl  = attrs.get('name')
    clf = '::' + attrs['fullname']
    cls = attrs['fullname']
    clt = string.translate(str(clf), self.transtable)
    bases = self.getBases( attrs['id'] )
    if 'members' in attrs : members = string.split(attrs['members'])
    mod = self.genModifier(attrs,None)
    typ = '::Reflex::' + self.xref[attrs['id']]['elem'].upper()
    if attrs.has_key('abstract') : mod += ' | ::Reflex::ABSTRACT'
    if self.vtables :
      if attrs['id'] in self.vtables : mod += ' | ::Reflex::VIRTUAL'
    else :  # new in version 0.6.0
      if self.isClassVirtual(attrs) :  mod += ' | ::Reflex::VIRTUAL'
    members = filter(self.memberfilter, members)  # Eliminate problematic members

    # Fill the different streams sc: constructor, ss: stub functions
    sc = ''

    #-------------------------------------------------------------------------------
    # Get the data members information and write down the schema evolution functions
    #-------------------------------------------------------------------------------
    memberTypeMap = self.createTypeMap( members )
    if ioReadRules:
      self.removeBrokenIoRules( cls, ioReadRules, memberTypeMap );
      sc += self.processIoReadFunctions( cls, clt, ioReadRules, memberTypeMap )
    if ioReadRawRules:
      self.removeBrokenIoRules( csl, ioReadRawRules, memberTypeMap );
      sc += self.processIoReadRawFunctions( cls, clt, ioReadRawRules, memberTypeMap )

    sc += '//------Dictionary for class %s -------------------------------\n' % cl
    sc += 'void %s_db_datamem(Reflex::Class*);\n' % (clt,)
    sc += 'void %s_db_funcmem(Reflex::Class*);\n' % (clt,)
    sc += 'Reflex::GenreflexMemberBuilder %s_datamem_bld(&%s_db_datamem);\n' % (clt, clt)
    sc += 'Reflex::GenreflexMemberBuilder %s_funcmem_bld(&%s_db_funcmem);\n' % (clt, clt)
    sc += 'void %s_dict() {\n' % (clt,)

    # Write the schema evolution rules
    if ioReadRules or ioReadRawRules:
      sc += '  ROOT::TSchemaHelper* rule;\n'
      
    if ioReadRules:
      sc += '  // the io read rules\n'
      sc += '  std::vector<ROOT::TSchemaHelper> readrules(%d);\n' % (len(ioReadRules),)
      sc += self.processIoRules( ioReadRules, 'readrules' )
      sc += '\n\n'

    if ioReadRawRules:
      sc += '  // the io readraw rules\n'
      sc += '  std::vector<ROOT::TSchemaHelper> readrawrules(%d);\n' % (len(ioReadRawRules),)
      sc += self.processIoRules( ioReadRawRules, 'readrawrules' )
      sc += '\n\n'

    if 'extra' in attrs and 'contid' in attrs['extra'] : 
      cid = attrs['extra']['contid'].upper()
    else :
      cid = getContainerId(clf)[0]
    notAccessibleType = self.checkAccessibleType(self.xref[attrs['id']])
    
    if self.isUnnamedType(clf) : 
      sc += '  ::Reflex::ClassBuilder(Reflex::Literal("%s"), typeid(::Reflex::Unnamed%s), sizeof(%s), %s, %s)' % ( cls, self.xref[attrs['id']]['elem'], '__shadow__::'+ string.translate(str(clf),self.transtable), mod, typ )
    elif notAccessibleType :
      sc += '  ::Reflex::ClassBuilder(Reflex::Literal("%s"), typeid(%s%s), sizeof(%s), %s, %s)' % ( cls, '::Reflex::' + self.xref[notAccessibleType]['attrs']['access'].title(), self.xref[attrs['id']]['elem'], '__shadow__::'+ string.translate(str(clf),self.transtable), mod, typ )
    else :
      typeidtype = '::' + cls
      # a funny bug in MSVC7.1: sizeof(::namesp::cl) doesn't work
      if sys.platform == 'win32':
         typeidtype = 'MSVC71_typeid_bug_workaround'
         sc += '  typedef ::%s %s;\n' % (cls, typeidtype)
      sc += '  ::Reflex::ClassBuilder(Reflex::Literal("%s"), typeid(%s), sizeof(::%s), %s, %s)' \
            % (cls, typeidtype, cls, mod, typ)
    if 'extra' in attrs :
      for pname, pval in attrs['extra'].items() :
        if pname not in ('name','pattern','n_name','file_name','file_pattern','fields') :
          if pname == 'id' : pname = 'ClassID'
          if pval[:5] == '!RAW!' :
            sc += '\n  .AddProperty(Reflex::Literal("%s"), %s)' % (pname, pval[5:])
          else :
            sc += '\n  .AddProperty(Reflex::Literal("%s"), "%s")' % (pname, pval)

    if ioReadRules:
      sc += '\n  .AddProperty("ioread", readrules )'
    if ioReadRawRules:
      sc += '\n  .AddProperty("ioreadraw", readrawrules )'

    for b in bases :
      sc += '\n' + self.genBaseClassBuild( clf, b )

    # on demand builder:
    # data member, prefix
    odbdp = '//------Delayed data member builder for class %s -------------------\n' % cl
    odbd = ''
    # function member, prefix
    odbfp = '//------Delayed function member builder for class %s -------------------\n' % cl
    odbf = ''

    for m in members :
      funcname = 'gen'+self.xref[m]['elem']+'Build'
      if funcname in dir(self) :
        line = self.__class__.__dict__[funcname](self, self.xref[m]['attrs'], self.xref[m]['subelems'])
        if line :
          if not self.xref[m]['attrs'].get('artificial') in ('true', '1') :
            if funcname == 'genFieldBuild' :
              odbd += '\n' + line
            elif funcname in ('genMethodBuild','genOperatorMethodBuild','genConverterBuild') : # put c'tors and d'tors into non-delayed part
              odbf += '\n' + line
            else :
              sc += '\n' + line
          else :
            sc += '\n' + line
    if len(odbd) :
      sc += '\n  .AddOnDemandDataMemberBuilder(&%s_datamem_bld)' % (clt)
      odbdp += 'void %s_db_datamem(Reflex::Class* cl) {\n' % (clt,)
      odbdp += '  ::Reflex::ClassBuilder(cl)'
      odbd += ';'
    else :
      odbdp += 'void %s_db_datamem(Reflex::Class*) {\n' % (clt,)
    if len(odbf) :
      sc += '\n  .AddOnDemandFunctionMemberBuilder(&%s_funcmem_bld)' % (clt)
      odbfp += 'void %s_db_funcmem(Reflex::Class* cl) {\n' % (clt,)
      odbfp += '  ::Reflex::ClassBuilder(cl)'
      odbf += ';'
    else :
      odbfp += 'void %s_db_funcmem(Reflex::Class*) {\n' % (clt,)
    sc += ';\n}\n\n'

    sc += odbdp + odbd + '\n}\n'
    sc += odbfp + odbf + '\n}\n'

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
    while type['elem'] in ('PointerType','Typedef','ArrayType') :
      type = self.xref[type['attrs']['type']]
    attrs = type['attrs']
    if 'access' in attrs and attrs['access'] in ('private','protected') : return attrs['id']
    if 'context' in attrs and self.checkAccessibleType(self.xref[attrs['context']]) : return attrs['id']
    return 0
#----------------------------------------------------------------------------------
  def funPtrPos(self, name) :
    if name.find(')(') != -1 :
      opp = 0
      clp = 0
      pos = -2
      for str in name.split(')('):
        opp += str.count('<')
        clp += str.count('>')
        pos += len(str) + 2
        if ( opp == clp ) : return pos
    return 0
#----------------------------------------------------------------------------------
  def genClassShadow(self, attrs, inner = 0 ) :
    if not inner :
      if attrs['id'] in self.generated_shadow_classes : return ''
      else : self.generated_shadow_classes.append(attrs['id'])
    inner_shadows = {}
    bases = self.getBases( attrs['id'] )
    if inner and attrs.has_key('demangled') and self.isUnnamedType(attrs['demangled']) :
      cls = attrs['demangled']
      clt = ''
    else:
      cls = self.genTypeName(attrs['id'],const=True,colon=True)
      clt = string.translate(str(cls), self.transtable)
    if clt :
      c = '#ifdef ' + clt + '\n' + '#undef ' + clt + '\n' + '#endif' + '\n'
    else :
      c = ''
    xtyp = self.xref[attrs['id']]
    typ = xtyp['elem'].lower()
    indent = inner * 2 * ' '
    if typ == 'enumeration' :
      c += indent + 'enum %s {};\n' % clt
    else:
      if not bases : 
        c += indent + '%s %s {\n%s  public:\n' % (typ, clt, indent)
      else :
        c += indent + '%s %s : ' % (typ, clt)
        for b in bases :
          if b.get('virtual','') == '1' : acc = 'virtual ' + b['access']
          else                          : acc = b['access']
          bname = self.genTypeName(b['type'],colon=True)
          if self.xref[b['type']]['attrs'].get('access') in ('private','protected'):
            bname = string.translate(str(bname),self.transtable)
            if not inner: c = self.genClassShadow(self.xref[b['type']]['attrs']) + c
          c += indent + '%s %s' % ( acc , bname )
          if b is not bases[-1] : c += ', ' 
        c += indent + ' {\n' + indent +'  public:\n'
      if clt: # and not self.checkAccessibleType(xtyp):
        c += indent + '  %s();\n' % (clt)
        if self.isClassVirtual( attrs ) :
          c += indent + '  virtual ~%s() throw();\n' % ( clt )
      members = attrs.get('members','')
      memList = members.split()
      # Inner class/struct/union/enum.
      for m in memList :
        member = self.xref[m]
        if member['elem'] in ('Class','Struct','Union','Enumeration') \
           and member['attrs'].get('access') in ('private','protected') \
           and not self.isUnnamedType(member['attrs'].get('demangled')):
          cmem = self.genTypeName(member['attrs']['id'],const=True,colon=True)
          if cmem != cls and cmem not in inner_shadows :
            inner_shadows[cmem] = string.translate(str(cmem), self.transtable)
            c += self.genClassShadow(member['attrs'], inner + 1)
            
      #
      # Virtual methods, see https://savannah.cern.ch/bugs/index.php?32874
      # Shadow classes inherit from the same bases as the shadowed class; if a
      # shadowed class is inherited from at least two bases and it defines
      # virtual methods of at least two bases then these virtual methods must
      # be declared in the shadow class or the compiler will complain about
      # ambiguous inheritance.
      allbases = []
      self.getAllBases(attrs['id'], allbases)
      if len(allbases) > 1 :
        allBasesMethods = {}
        # count method occurrences collected over all bases
        for b in allbases:
          baseattrs = self.xref[b[0]]['attrs']
          currentBaseName = baseattrs['demangled']
          basemem = baseattrs.get('members','')
          basememList = members.split()
          for bm in basememList:
            basemember = self.xref[bm]
            if basemember['elem'] in ('Method','OperatorMethod') \
                 and basemember['attrs'].get('virtual') == '1' \
                 and self.isTypePublic(basemember['attrs']['returns']) \
                 and not self.hasNonPublicArgs(basemember['subelems']):
              # This method is virtual and publicly accessible.
              # Remove the class name and the scope operator from the demangled method name.
              demangledBaseMethod = basemember['attrs'].get('demangled')
              posFuncName = demangledBaseMethod.rfind('::' + basemember['attrs'].get('name') + '(')
              if posFuncName == -1 : continue
              demangledBaseMethod = demangledBaseMethod[posFuncName + 2:]
              found = 0
              if demangledBaseMethod in allBasesMethods.keys():
                # the method exists in another base.
                # getAllBases collects the bases along each line of inheritance,
                # i.e. either the method we found is in a derived class of b
                # or it's in a different line and we have to write it out
                # to prevent ambiguous inheritance.
                for foundbases in allBasesMethods[demangledBaseMethod]['bases']:
                  if b in foundbases:
                    found = 1
                    break
                if found == 0: found = 2
              if found != 1:
                allbasebases = []
                self.getAllBases(baseattrs['id'], allbasebases)
                if found == 0:
                  allBasesMethods[demangledBaseMethod] = { 'bases': ( allbasebases ), 'returns': basemember['attrs'].get('returns') }
                else:
                  allBasesMethods[demangledBaseMethod]['bases'].append( allbasebases )
                  allBasesMethods[demangledBaseMethod]['returns'] = basemember['attrs'].get('returns')
        # write out ambiguous methods
        for demangledMethod in allBasesMethods.keys() :
          member = allBasesMethods[demangledMethod]
          if len(member['bases']) > 1:
            ret = self.genTypeName(member['returns'], enum=False, const=False, colon=True)
            if '(' not in ret:
              # skip functions returning functions; we don't get the prototype right easily:
              cmem = '  virtual %s %s throw();' % (ret, demangledMethod)
              c += indent + cmem + '\n'
      # Data members.
      for m in memList :
        member = self.xref[m]
        if member['elem'] in ('Field',) :
          a = member['attrs']
          axref = self.xref[a['type']]
          t = self.genTypeName(a['type'],colon=True,const=True)
          arraytype = ""
          if t[-1] == ']' : arraytype = t[t.find('['):]

          fundtype = axref
          while fundtype['elem'] in ('ArrayType', 'Typedef'):
            fundtype = self.xref[fundtype['attrs']['type']]
          mTypeElem = fundtype['elem']

          #---- Check if pointer of reference - exact type irrelevant
          if mTypeElem == 'PointerType' :
            c += indent + '  void* %s;\n' % (a['name'] + arraytype)
            continue
          elif mTypeElem ==  'ReferenceType' :
            c += indent + '  int& %s;\n' % (a['name'] + arraytype)
            continue

          #---- Check if a type and a member with the same name exist in the same scope
          if mTypeElem in ('Class','Struct'):
            mTypeName = fundtype['attrs'].get('name')
            mTypeId = fundtype['attrs']['id']
            for el in self.xref[fundtype['attrs']['context']]['attrs'].get('members').split():
              if self.xref[el]['attrs'].get('name') == mTypeName and mTypeId != el :
                t = mTypeElem.lower() + ' ' + t[2:]
                break
          #---- Check for non public types------------------------
          noPublicType = self.checkAccessibleType(axref)
          if ( noPublicType and not self.isUnnamedType(axref['attrs'].get('demangled'))):
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
          mType = axref
          if mType and self.isUnnamedType(mType['attrs'].get('demangled')) :
            t = self.genClassShadow(mType['attrs'], inner+1)[:-2]
          fPPos = self.funPtrPos(t)
          if t[-1] == ']'         : c += indent + '  %s %s;\n' % ( t[:t.find('[')], a['name'] + arraytype )
          elif fPPos              : c += indent + '  %s;\n'    % ( t[:fPPos] + a['name'] + t[fPPos:] )
          else                    : c += indent + '  %s %s;\n' % ( t, a['name'] )
      c += indent + '};\n'
    return c    
#----------------------------------------------------------------------------------
  def genTypedefBuild(self, attrs, childs) :
    if self.no_membertypedefs : return ''
    # access selection doesn't work with gccxml0.6 - typedefs don't have it
    if self.interpreter and 'access' in attrs : return ''
    s = ''
    s += '  .AddTypedef(%s, Reflex::Literal("%s::%s"))' % ( self.genTypeID(attrs['type']), self.genTypeName(attrs['context']), attrs['name']) 
    return s  
#----------------------------------------------------------------------------------
  def genEnumerationBuild(self, attrs, childs):
    s = ''
    name = self.genTypeName(attrs['id']) 
    values = ''
    for child in childs : values += child['name'] + '=' + child['init'] +';'
    values = values[:-1]
    mod = self.genModifier(attrs, None)
    if self.isUnnamedType(name) :
      s += '  .AddEnum(Reflex::Literal("%s"), Reflex::Literal("%s"), &typeid(::Reflex::UnnamedEnum), %s)' % (name[name.rfind('::')+3:], values, mod) 
    else :
      if attrs.get('access') in ('protected','private'):
        if not self.interpreter:
          s += '  .AddEnum(Reflex::Literal("%s"), Reflex::Literal("%s"), &typeid(::Reflex::UnknownType), %s)' % (name, values, mod)        
      else:
        s += '  .AddEnum(Reflex::Literal("%s"), Reflex::Literal("%s"), &typeid(%s), %s)' % (name, values, name, mod)
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
# const is CONST VETO!!!
  def genTypeName(self, id, enum=False, const=False, colon=False, alltempl=False, _useCache=True, _cache={}) :
    if _useCache:
      key = (self,id,enum,const,colon,alltempl)
      if _cache.has_key(key):
        return _cache[key]
      else:
        ret = self.genTypeName(id,enum,const,colon,alltempl,False)
        _cache[key] = ret
        return ret
    elem  = self.xref[id]['elem']
    attrs = self.xref[id]['attrs']
    if self.isUnnamedType(attrs.get('demangled')) :
      if colon : return '__'+attrs['demangled']
      else : return attrs['demangled']
    if id[-1] in ['c','v'] :
      nid = id[:-1]
      if nid[-1] in ['c','v'] :
        nid = nid[:-1]
      cvdict = {'c':'const','v':'volatile'}
      prdict = {'PointerType':'*', 'ReferenceType':'&'}
      nidelem = self.xref[nid]['elem']
      if nidelem in ('PointerType','ReferenceType') :
        if const : return self.genTypeName(nid, enum, False, colon)
        else :     return self.genTypeName(nid, enum, False, colon) + ' ' + cvdict[id[-1]]
      else :
        if const : return self.genTypeName(nid, enum, False, colon)
        else     : return cvdict[id[-1]] + ' ' + self.genTypeName(nid, enum, False, colon)
    # "const" vetoeing must not recurse
    const = False
    s = self.genScopeName(attrs, enum, const, colon)
    if elem == 'Namespace' :
      if 'name' not in attrs : s += '@anonymous@namespace@'
      elif attrs['name'] != '::' : s += attrs['name']
    elif elem == 'PointerType' :
      t = self.genTypeName(attrs['type'],enum, const, colon)
      if   t[-1] == ')' or t[-7:] == ') const' or t[-10:] == ') volatile' : s += t.replace('::*)','::**)').replace('::)','::*)').replace('(*)', '(**)').replace('()','(*)')
      elif t[-1] == ')' or t[-7:] == ') const' or t[-10:] == ') volatile' : s += t[:t.find('[')] + '(*)' + t[t.find('['):]
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
      max = attrs['max'].rstrip('u')
      arr = '[]'
      if len(max):
        arr = '[%s]' % str(int(max)+1)
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
      # OffsetType A::*, different treatment for GCCXML 0.7 and 0.9:
      # 0.7: basetype: A*
      # 0.9: basetype: A - add a "*" here
      version = float(re.compile('\\b\\d+\\.\\d+\\b').match(self.gccxmlvers).group())
      if  version >= 0.9 : 
        s += "*"
    else :
      if 'name' in attrs : s += attrs['name']
      s = normalizeClass(s,alltempl,_useCache=_useCache) # Normalize STL class names, primitives, etc.
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
        if 'returns' in attrs : c = '::Reflex::FunctionTypeBuilder(' + self.genTypeID(attrs['returns'])
        else                  : c = '::Reflex::FunctionTypeBuilder(type_void'
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
    self.typeids += self.fktypeids
    # l: literals, d: derived
    l = ['  ::Reflex::Type type_void = ::Reflex::TypeBuilder(Reflex::Literal("void"));\n']
    d = ''
    for id in self.typeids :
      n = '  ::Reflex::Type type%s = ' % id
      if id[-1] == 'c':
        d += n + '::Reflex::ConstBuilder(type'+id[:-1]+');\n'
      elif id[-1] == 'v':
        d += n + '::Reflex::VolatileBuilder(type'+id[:-1]+');\n'
      else : 
        elem  = self.xref[id]['elem']
        attrs = self.xref[id]['attrs']
        if elem == 'PointerType' :
          d += n + '::Reflex::PointerBuilder(type'+attrs['type']+');\n'
        elif elem == 'ReferenceType' :
          d += n + '::Reflex::ReferenceBuilder(type'+attrs['type']+');\n'
        elif elem == 'ArrayType' :
          mx = attrs['max'].rstrip('u')
          # check if array is bound (max='fff...' for unbound arrays)
          if mx.isdigit() : alen = str(int(mx)+1)
          else            : alen = '0' 
          d += n + '::Reflex::ArrayBuilder(type'+attrs['type']+', '+ alen +');\n'
        elif elem == 'Typedef' :
          sc = self.genTypeName(attrs['context'])
          if sc : sc += '::'
          d += n + '::Reflex::TypedefTypeBuilder(Reflex::Literal("'+sc+attrs['name']+'"), type'+ attrs['type']+');\n'
        elif elem == 'OffsetType' :
          l.append(n + '::Reflex::TypeBuilder(Reflex::Literal("%s"));\n' % self.genTypeName(attrs['id']))
        elif elem == 'FunctionType' :
          if 'returns' in attrs : d += n + '::Reflex::FunctionTypeBuilder(type'+attrs['returns']
          else                  : d += n + '::Reflex::FunctionTypeBuilder(type_void'
          args = self.xref[id]['subelems']
          for a in args : d += ', type'+ a['type']
          d += ');\n'
        elif elem == 'MethodType' :
          l.append(n + '::Reflex::TypeBuilder(Reflex::Literal("%s"));\n' % self.genTypeName(attrs['id']))
        elif elem in ('OperatorMethod', 'Method', 'Constructor', 'Converter', 'Destructor',
                      'Function', 'OperatorFunction') :
          pass
        elif elem == 'Enumeration' :
          sc = self.genTypeName(attrs['context'])
          if sc : sc += '::'
          # items = self.xref[id]['subelems']
          # values = string.join([ item['name'] + '=' + item['init'] for item in items],';"\n  "')          
          #c += 'EnumTypeBuilder("' + sc + attrs['name'] + '", "' + values + '");\n'
          l.append(n + '::Reflex::EnumTypeBuilder(Reflex::Literal("' + sc + attrs['name'] + '"));\n')
        else :
         name = ''
         if 'name' not in attrs and 'demangled' in attrs : name = attrs.get('demangled')
         else:
           if 'context' in attrs :
             ns = self.genTypeName(attrs['context'])
             if ns : name += ns + '::'
           if 'name' in attrs :
             name += attrs['name']
         name = normalizeClass(name,False)
         l.append(n + '::Reflex::TypeBuilder(Reflex::Literal("'+name+'"));\n')
    #def lenCmp(a,b): return cmp(len(str(a)), len(str(b)))
    l.sort(key=lambda i: len(i))
    return ''.join(l) + d
#----------------------------------------------------------------------------------
  def genNamespaces(self, selected ) :
    used_context = []
    s = ''
    for c in selected :
      if 'incomplete' not in c : used_context.append(c['context'])
    idx = 0
    for ns in self.namespaces :
      if ns['id'] in used_context and 'name' in ns and  ns['name'] != '::' :
        s += '  ::Reflex::NamespaceBuilder nsb%d( Reflex::Literal("%s") );\n' % (idx, self.genTypeName(ns['id']))
        idx += 1
    return s
#----------------------------------------------------------------------------------
  def genFunctionsStubs(self, selfunctions) :
    s = ''
    for f in selfunctions :
      id   = f['id']
      name = self.genTypeName(id)
      self.genTypeID(id)
      args = self.xref[id]['subelems']
      returns  = self.genTypeName(f['returns'], enum=True, const=True)
      demangled = self.xref[id]['attrs'].get('demangled')
      if not demangled or not len(demangled):
        demangled = name
      if not self.quiet : print  'function '+ demangled
      retaddrpar=''
      if returns != 'void': retaddrpar=' retaddr'
      argspar=''
      if len(args) : argspar=' arg'
      head =  'static void function%s( void*%s, void*, const std::vector<void*>&%s, void*)\n{\n' % (id, retaddrpar, argspar)
      ndarg = self.getDefaultArgs(args)
      narg  = len(args)
      if ndarg : iden = '  '
      else     : iden = ''
      body = ''
      for n in range(narg-ndarg, narg+1) :
        if ndarg :
          if n == narg-ndarg :  body += '  if ( arg.size() == %d ) {\n' % n
          else               :  body += '  else if ( arg.size() == %d ) { \n' % n
        if returns == 'void' :
          body += iden + '  %s(' % ( name, )
          head, body = self.genMCOArgs(args, n, len(iden)+2, head, body)
          body += ');\n'
        else :
          if returns[-1] in ('*',')' ) and returns.find('::*') == -1:
            body += iden + '  if (retaddr) *(void**)retaddr = (void*)%s(' % ( name, )
            head, body = self.genMCOArgs(args, n, len(iden)+2, head, body)
            body += ');\n'
            body += iden + '  else %s(' % ( name, )
            head, body = self.genMCOArgs(args, n, len(iden)+2, head, body)
            body += ');\n'
          elif returns[-1] == '&' :
            body += iden + '  if (retaddr) *(void**)retaddr = (void*)&%s(' % ( name, )
            head, body = self.genMCOArgs(args, n, len(iden)+2, head, body)
            body += ');\n'
            # The seemingly useless '&' below is to work around Microsoft's
            # compiler 7.1-9 odd complaint C2027 if the reference has only
            # been forward declared.
            if sys.platform == 'win32':
              body += iden + '  else &%s(' % ( name, )
            else:
              # but '&' will trigger an "unused value" warning on != MSVC
              body += iden + '  else %s(' % ( name, )
            head, body = self.genMCOArgs(args, n, len(iden)+2, head, body)
            body += ');\n'
          else :
            body += iden + '  if (retaddr) new (retaddr) (%s)(%s(' % ( returns, name )
            head, body = self.genMCOArgs(args, n, len(iden)+2, head, body)
            body += '));\n'
            body += iden + '  else %s(' % ( name )
            head, body = self.genMCOArgs(args, n, len(iden)+2, head, body)
            body += ');\n'
        if ndarg : 
          if n != narg : body += '  }\n'
          else :
            if returns == 'void' : body += '  }\n'
            else :                 body += '  }\n'
      s += head + body + '}\n'
    return s  
#----------------------------------------------------------------------------------
  def genFunctions(self, selfunctions) :
    s = ''
    i = 0;
    for f in selfunctions :
      id   = f['id']
      name = self.genTypeName(id)
      args = self.xref[id]['subelems']      
      if args : params  = '"'+ string.join( map(self.genParameter, args),';')+'"'
      else    : params  = '0'
      mod = self.genModifier(f, None)
      s += '      ::Reflex::Type t%s = %s;' % (i, self.genTypeID(id))
      s += '      ::Reflex::FunctionBuilder(t%s, Reflex::Literal("%s"), function%s, 0, Reflex::Literal(%s), %s);\n' % (i, name, id, params, mod)
      i += 1;
    return s
#----------------------------------------------------------------------------------
  def genEnums(self, selenums) :
    s = ''
    i = 0;
    for e in selenums :
      # Do not generate dictionaries for unnamed enums; we cannot reference them anyway.
      if not e.has_key('name') or len(e['name']) == 0 or e['name'][0] == '.':
        continue
      id   = e['id']
      cname = self.genTypeName(id, colon=True)
      name  = self.genTypeName(id)
      mod = self.genModifier(self.xref[id]['attrs'], None)
      if not self.quiet : print 'enum ' + name
      s += '      ::Reflex::EnumBuilder(Reflex::Literal("%s"),typeid(%s), %s)' % (name, cname, mod)
      items = self.xref[id]['subelems']
      for item in items :
        s += '\n        .AddItem(Reflex::Literal("%s"),%s)' % (item['name'], item['init'])
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
      s += '      ::Reflex::VariableBuilder(Reflex::Literal("%s"), %s, (size_t)&%s, %s );\n' % (name, self.genTypeID(v['type']),self.genTypeName(id), mod)
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
    if self.selector :
      fieldsel = self.selector.matchfield(cls,name)
      if not fieldsel[1]: xattrs = fieldsel[0]
      else              : return ""
    else             : xattrs = None
    mod = self.genModifier(attrs,xattrs)
    if attrs['type'][-1] == 'c' :
      if mod : mod += ' | ::Reflex::CONST'
      else   : mod =  '::Reflex::CONST'
    if attrs['type'][-1] == 'v' :
      if mod : mod += ' | ::Reflex::VOLATILE'
      else   : mod = '::Reflex::VOLATILE'
    shadow = '__shadow__::' + string.translate( str(cl), self.transtable)
    c = '  .AddDataMember(%s, Reflex::Literal("%s"), OffsetOf(%s, %s), %s)' % (self.genTypeID(attrs['type']), name, shadow, name, mod)
    c += self.genCommentProperty(attrs)
    # Other properties
    if xattrs : 
      for pname, pval in xattrs.items() : 
        if pname not in ('name', 'transient', 'pattern') :
          c += '\n  .AddProperty(Reflex::Literal("%s"),Reflex::Literal("%s"))' % (pname, pval)     
    return c
#----------------------------------------------------------------------------------
  def genVariableBuild(self, attrs, childs):
    if 'access' in attrs and attrs['access'] in ('private','protected') : return ''
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
    if mod : mod += ' | Reflex::STATIC'
    else   : mod =  'Reflex::STATIC'
    if attrs['type'][-1] == 'c' :
      if mod : mod += ' | Reflex::CONST'
      else   : mod =  'Reflex::CONST'
    if attrs['type'][-1] == 'v' :
      if mod : mod += ' | Reflex::VOLATILE'
      else   : mod = 'Reflex::VOLATILE'
    c = ''
    if not attrs.has_key('init'):
      c = '  .AddDataMember(%s, Reflex::Literal("%s"), (size_t)&%s::%s, %s)' % (self.genTypeID(attrs['type']), name, cls, name, mod)
      c += self.genCommentProperty(attrs)
      # Other properties
      if xattrs : 
        for pname, pval in xattrs.items() : 
          if pname not in ('name', 'transient', 'pattern') :
            c += '\n  .AddProperty(Reflex::Literal("%s"),Reflex::Literal("%s"))' % (pname, pval)     
    return c
#----------------------------------------------------------------------------------    
  def genCommentProperty(self, attrs):
    if not (self.comments or self.iocomments) \
       or 'file' not in attrs \
       or ('artificial' in attrs and attrs['artificial'] == '1') : return '' 
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
    poscomment = line.find('//')
    if poscomment == -1 : return ''
    if not self.comments and self.iocomments:
      if line[poscomment+2] != '!' \
         and line[poscomment+2] != '[' \
         and line[poscomment+2:poscomment+4] != '->' \
         and line[poscomment+2:poscomment+4] != '||': return ''
    return '\n  .AddProperty("comment",Reflex::Literal("%s"))' %  (line[poscomment+2:-1]).replace('"','\\"')
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
    if   attrs.get('access') == 'public' or 'access' not in attrs : mod = '::Reflex::PUBLIC'
    elif attrs['access'] == 'private'   : mod = '::Reflex::PRIVATE'
    elif attrs['access'] == 'protected' : mod = '::Reflex::PROTECTED'
    else                                : mod = '::Reflex::NONE'
    if 'virtual' in attrs : mod += ' | ::Reflex::VIRTUAL'
    if 'pure_virtual' in attrs : mod += ' | ::Reflex::ABSTRACT'
    if 'static'  in attrs : mod += ' | ::Reflex::STATIC'
    # Extra modifiers
    xtrans = ''
    etrans = ''
    if xattrs :
      xtrans = xattrs.get('transient')
      if xtrans : xtrans = xtrans.lower()
    if 'extra' in attrs:
      etrans = attrs['extra'].get('transient')
      if etrans : etrans = etrans.lower()
    if xtrans == 'true' or etrans == 'true' : mod += ' | ::Reflex::TRANSIENT'
    if 'artificial' in attrs : mod += ' | ::Reflex::ARTIFICIAL' 
    if 'explicit' in attrs : mod += ' | ::Reflex::EXPLICIT'
    if 'mutable' in attrs : mod += ' | ::Reflex::MUTABLE' 
    return mod
#----------------------------------------------------------------------------------
  def genMCODecl( self, type, name, attrs, args ) :
    static = 'static '
    if sys.platform == 'win32' and type in ('constructor', 'destructor'): static = ''
    return static + 'void %s%s(void*, void*, const std::vector<void*>&, void*);' % (type, attrs['id'])
#----------------------------------------------------------------------------------
  def genMCOBuild(self, type, name, attrs, args):
    id       = attrs['id']
    if self.isUnnamedType(self.xref[attrs['context']]['attrs'].get('demangled')) or \
       self.checkAccessibleType(self.xref[attrs['context']]) : return ''
    if type == 'constructor' : returns  = 'void'
    else                     : returns  = self.genTypeName(attrs['returns'])
    mod = self.genModifier(attrs, None)
    if   type == 'constructor' : mod += ' | ::Reflex::CONSTRUCTOR'
    elif type == 'operator' :    mod += ' | ::Reflex::OPERATOR'
    elif type == 'converter' :   mod += ' | ::Reflex::CONVERTER'
    if attrs.get('const')=='1' : mod += ' | ::Reflex::CONST'
    if args : params  = '"'+ string.join( map(self.genParameter, args),';')+'"'
    else    : params  = '0'
    s = '  .AddFunctionMember(%s, Reflex::Literal("%s"), %s%s, 0, %s, %s)' % (self.genTypeID(id), name, type, id, params, mod)
    s += self.genCommentProperty(attrs)
    return s
#----------------------------------------------------------------------------------
  def genMCODef(self, type, name, attrs, args):
    id       = attrs['id']
    cl       = self.genTypeName(attrs['context'],colon=True)
    clt      = string.translate(str(cl), self.transtable)
    returns  = self.genTypeName(attrs['returns'],enum=True, const=True)
    narg     = len(args)
    argspar  = ''
    if narg : argspar = ' arg'
    retaddrpar = ''

    # If we construct a conversion operator to pointer to function member the name
    # will contain TDF_<attrs['id']>
    tdfname = 'TDF%s'%attrs['id']
    tdfdecl = ''
    if name.find(tdfname) != -1 :
      tdfdecl = '  typedef %s;\n'%name
      name = 'operator ' + tdfname
      returns = tdfname

    if returns != 'void': retaddrpar=' retaddr'

    static = 'static '
    if sys.platform == 'win32' and type in ('constructor', 'destructor'): static = ''
    head =  '%s void %s%s( void*%s, void* o, const std::vector<void*>&%s, void*)\n{\n' %( static, type, id, retaddrpar, argspar )
    head += tdfdecl
    ndarg = self.getDefaultArgs(args)
    if ndarg : iden = '  '
    else     : iden = ''
    if 'const' in attrs : cl = 'const '+ cl
    body = ''
    for n in range(narg-ndarg, narg+1) :
      if ndarg :
        if n == narg-ndarg :  body += '  if ( arg.size() == %d ) {\n' % n
        else               :  body += '  else if ( arg.size() == %d ) { \n' % n
      if returns != 'void' :
        if returns[-1] in ('*',')') and returns.find('::*') == -1 :
          body += iden + '  if (retaddr) *(void**)retaddr = Reflex::FuncToVoidPtr((((%s*)o)->%s)(' % ( cl, name )
          head, body = self.genMCOArgs(args, n, len(iden)+2, head, body)
          body += '));\n' + iden + '  else '
        elif returns[-1] == '&' :
          body += iden + '  if (retaddr) *(void**)retaddr = (void*)&(((%s*)o)->%s)(' % ( cl, name )
          head, body = self.genMCOArgs(args, n, len(iden)+2, head, body)
          body += ');\n' + iden + '  else '
        else :
          body += iden + '  if (retaddr) new (retaddr) (%s)((((%s*)o)->%s)(' % ( returns, cl, name )
          head, body = self.genMCOArgs(args, n, len(iden)+2, head, body)
          body += '));\n' + iden + '  else '
      if returns[-1] == '&' :
        # The seemingly useless '&' below is to work around Microsoft's
        # compiler 7.1-9 odd complaint C2027 if the reference has only
        # been forward declared.
        if sys.platform == 'win32':
          body += iden + '  &(((%s*)o)->%s)(' % ( cl, name )
        else:
          # but '&' will trigger an "unused value" warning on != MSVC
          body += iden + '  (((%s*)o)->%s)(' % ( cl, name )
      else: 
        body += iden + '  (((%s*)o)->%s)(' % ( cl, name )
      head, body = self.genMCOArgs(args, n, len(iden)+2, head, body)
      body += ');\n'
      if ndarg : 
        if n != narg : body += '  }\n'
        else :
          if returns == 'void' : body += '  }\n'
          else :                 body += '  }\n'
    body += '}\n'
    return head + body;
#----------------------------------------------------------------------------------
  def getDefaultArgs(self, args):
    n = 0
    for a in args :
      if 'default' in a : n += 1
    return n
#----------------------------------------------------------------------------------
  def genMCOArgs(self, args, narg, pad, head, body):
    s = ''
    td = ''
    for i in range(narg) :
      a = args[i]
      #arg = self.genArgument(a, 0);
      arg = self.genTypeName(a['type'],colon=True)
      # Create typedefs to let us handle arrays, but skip arrays that are template parameters.
      if arg.find('[') != -1 and arg.count('<', 0, arg.find('[')) <= arg.count('>', 0, arg.find('[')):
        if arg[-1] == '*' :
          argnoptr = arg[:-1]
          argptr = '*'
        elif len(arg) > 7 and arg[-7:] == '* const':
          argnoptr = arg[:-7]
          argptr = '* const'
        else :
          argnoptr = arg
          argptr = ''
        if len(argnoptr) > 1 and argnoptr[-1] == '&':
          argnoptr = argnoptr[:-1]
        td += pad*' ' + 'typedef %s RflxDict_arg_td%d%s;\n' % (argnoptr[:argnoptr.index('[')], i, argnoptr[argnoptr.index('['):])
        arg = 'RflxDict_arg_td%d' % i
        arg += argptr;
      if arg[-1] == '*' or len(arg) > 7 and arg[-7:] == '* const':
        if arg[-2:] == ':*' or arg[-8:] == ':* const' : # Pointer to data member
          s += '*(%s*)arg[%d]' % (arg, i )
        else :
          s += '(%s)arg[%d]' % (arg, i )
      elif arg[-1] == ']' :
        s += '(%s)arg[%d]' % (arg, i)
      elif arg[-1] == ')' or (len(arg) > 7 and arg[-7:] == ') const'): # FIXME, the second check is a hack
        if arg.find('::*') != -1 :  # Pointer to function member
          s += '*(%s)arg[%d]' %(arg.replace('::*','::**'), i)
        elif (len(arg) > 7  and arg[-7:] == ') const') :
          s += 'Reflex::VoidPtrToFunc< %s >(arg[%d])' % (arg[:-6].replace('(*)','(* const)'), i) # 2nd part of the hack
        else :
          s += 'Reflex::VoidPtrToFunc< %s >(arg[%d])' % (arg, i )
      elif arg[-1] == '&' :
        s += '*(%s*)arg[%d]' % (arg[:-1], i )
      else :
        s += '*(%s*)arg[%d]' % (arg, i )
      if i != narg - 1 : s += ',\n' + (pad+2)*' '
    return head + td, body + s
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
    name = attrs.get('name')
    if not name : name = self.xref[attrs['context']]['attrs']['demangled'].split('::')[-1]
    return self.genMCOBuild( 'constructor', name, attrs, args )
#----------------------------------------------------------------------------------
  def genConstructorDef(self, attrs, args):
    cl  = self.genTypeName(attrs['context'], colon=True)
    clt = string.translate(str(cl), self.transtable)
    id  = attrs['id']
    paramargs = ''
    if len(args): paramargs = ' arg'
    head = ''
    if sys.platform != 'win32': head = 'static '
    head += 'void constructor%s( void* retaddr, void* mem, const std::vector<void*>&%s, void*) {\n' %( id, paramargs )
    body = ''
    if 'pseudo' in attrs :
      head += '  if (retaddr) *(void**)retaddr =  ::new(mem) %s( *(__void__*)0 );\n' % ( cl )
      head += '  else ::new(mem) %s( *(__void__*)0 );\n' % ( cl )
    else :
      ndarg = self.getDefaultArgs(args)
      narg  = len(args)
      for n in range(narg-ndarg, narg+1) :
        if ndarg :
          if n == narg-ndarg :  body += '  if ( arg.size() == %d ) {\n  ' % n
          else               :  body += '  else if ( arg.size() == %d ) { \n  ' % n
        body += '  if (retaddr) *(void**)retaddr = ::new(mem) %s(' % ( cl )
        head, body = self.genMCOArgs(args, n, 4, head, body)
        body += ');\n'
        body += '  else ::new(mem) %s(' % ( cl )
        head, body = self.genMCOArgs(args, n, 4, head, body)
        body += ');\n'
        if ndarg : 
          if n != narg : body += '  }\n'
          else :         body += '  }\n'
    body += '}\n'
    return head + body
#----------------------------------------------------------------------------------
  def genDestructorDef(self, attrs, childs):
    cl = self.genTypeName(attrs['context'])
    static = ''
    dtorscope = ''
    if sys.platform != 'win32':
        static = 'static '
        dtorscope = '::' + cl + '::'
    dtorimpl = '%svoid destructor%s(void*, void * o, const std::vector<void*>&, void *) {\n' % ( static, attrs['id'])
    if (attrs['name'][0] != '.'):
      return dtorimpl + '(((::%s*)o)->%s~%s)();\n}' % ( cl, dtorscope, attrs['name'] )
    else:
      # unnamed; can't call.
      return dtorimpl + '  // unnamed, cannot call destructor\n}'
#----------------------------------------------------------------------------------
  def genDestructorBuild(self, attrs, childs):
    if self.isUnnamedType(self.xref[attrs['context']]['attrs'].get('demangled')) or \
       self.checkAccessibleType(self.xref[attrs['context']]) : return ''
    mod = self.genModifier(attrs,None)
    id       = attrs['id']
    s = '  .AddFunctionMember(%s, Reflex::Literal("~%s"), destructor%s, 0, 0, %s | ::Reflex::DESTRUCTOR )' % (self.genTypeID(id), attrs['name'], attrs['id'], mod)
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
    return self.genMCOBuild( 'converter', 'operator '+self.genTypeName(attrs['returns'],enum=True,const=False), attrs, args )    
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
    mod = '::Reflex::' + b['access'].upper()
    if 'virtual' in b and b['virtual'] == '1' : mod = '::Reflex::VIRTUAL | ' + mod
    return '  .AddBase(%s, ::Reflex::BaseOffset< %s, %s >::Get(), %s)' %  (self.genTypeID(b['type']), clf, self.genTypeName(b['type'],colon=True), mod)
#----------------------------------------------------------------------------------
  def enhanceClass(self, attrs):
    if self.isUnnamedType(attrs.get('demangled')) or self.checkAccessibleType(self.xref[attrs['id']]) : return
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
        if (len(args) == 0  or 'default' in args[0] ) \
               and 'abstract' not in attrs \
               and self.xref[m]['attrs'].get('access') == 'public' \
               and not self.isDestructorNonPublic(attrs['id']):
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
    return 'static void method%s( void*, void*, const std::vector<void*>&, void* ); ' % (attrs['id'])
  def genCreateCollFuncTableBuild( self, attrs, args ) :
    mod = self.genModifier(attrs, None)
    return '  .AddFunctionMember<void*(void)>(Reflex::Literal("createCollFuncTable"), method%s, 0, 0, %s)' % ( attrs['id'], mod)
  def genCreateCollFuncTableDef( self, attrs, args ) :
    cl       = self.genTypeName(attrs['context'], colon=True)
    clt      = string.translate(str(cl), self.transtable)
    t        = getTemplateArgs(cl)[0]
    if cl[:13] == '::std::bitset'  :
      s  = 'static void method%s( void* retaddr, void*, const std::vector<void*>&, void*)\n{\n' %( attrs['id'], )
      s += '  if (retaddr) *(void**) retaddr = ::Reflex::Proxy< ::Reflex::StdBitSetHelper< %s > >::Generate();\n' % (cl,)
      s += '  else ::Reflex::Proxy< ::Reflex::StdBitSetHelper< %s > >::Generate();\n' % (cl,)
      s += '}\n'
    else:
      s  = 'static void method%s( void* retaddr, void*, const std::vector<void*>&, void*)\n{\n' %( attrs['id'], )
      s += '  if (retaddr) *(void**) retaddr = ::Reflex::Proxy< %s >::Generate();\n' % (cl,)
      s += '  else ::Reflex::Proxy< %s >::Generate();\n' % (cl,)
      s += '}\n'
    return s
#----BasesMap stuff--------------------------------------------------------
  def genGetBasesTableDecl( self, attrs, args ) :
    return 'static void method%s( void*, void*, const std::vector<void*>&, void* ); ' % (attrs['id'])
  def genGetBasesTableBuild( self, attrs, args ) :
    mod = self.genModifier(attrs, None)
    return '  .AddFunctionMember<void*(void)>(Reflex::Literal("__getBasesTable"), method%s, 0, 0, %s)' % (attrs['id'], mod)
  def genGetBasesTableDef( self, attrs, args ) :
    cid      = attrs['context']
    cl       = self.genTypeName(cid, colon=True)
    clt      = string.translate(str(cl), self.transtable)
    s  = 'static void method%s( void* retaddr, void*, const std::vector<void*>&, void*)\n{\n' %( attrs['id'], )
    s += '  typedef std::vector<std::pair< ::Reflex::Base, int> > Bases_t;\n'
    s += '  static Bases_t s_bases;\n'
    s += '  if ( !s_bases.size() ) {\n'
    bases = []
    self.getAllBases( cid, bases ) 
    for b in bases :
      bname = self.genTypeName(b[0],colon=True)
      bname2 = self.genTypeName(b[0])
      s += '    s_bases.push_back(std::make_pair(::Reflex::Base( ::Reflex::TypeBuilder(Reflex::Literal("%s")), ::Reflex::BaseOffset< %s,%s >::Get(),%s), %d));\n' % (bname2, cl, bname, b[1], b[2])
    s += '  }\n  if (retaddr) *(Bases_t**)retaddr = &s_bases;\n' 
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
    return 'static void method%s( void*, void*, const std::vector<void*>&, void* ); ' % (attrs['id'])
  def genGetNewDelFunctionsBuild( self, attrs, args ) :
    cid      = attrs['context']
    mod = self.genModifier(attrs, None)  
    return '  .AddFunctionMember<void*(void)>(Reflex::Literal("__getNewDelFunctions"), method_newdel%s, 0, 0, %s)' % (cid, mod)
  def genGetNewDelFunctionsDef( self, attrs, args ) :
    cid      = attrs['context']
    cl       = self.genTypeName(cid, colon=True)
    clt      = string.translate(str(cl), self.transtable)
    (newc, newa) = self.checkOperators(cid)
    s  = 'static void method_newdel%s( void* retaddr, void*, const std::vector<void*>&, void*)\n{\n' %( cid )
    s += '  static ::Reflex::NewDelFunctions s_funcs;\n'
    s += '  s_funcs.fNew         = ::Reflex::NewDelFunctionsT< %s >::new%s_T;\n' % (cl, newc)
    s += '  s_funcs.fNewArray    = ::Reflex::NewDelFunctionsT< %s >::newArray%s_T;\n' % (cl, newa)
    s += '  s_funcs.fDelete      = ::Reflex::NewDelFunctionsT< %s >::delete_T;\n' % cl
    s += '  s_funcs.fDeleteArray = ::Reflex::NewDelFunctionsT< %s >::deleteArray_T;\n' % cl
    s += '  s_funcs.fDestructor  = ::Reflex::NewDelFunctionsT< %s >::destruct_T;\n' % cl
    s += '  if (retaddr) *(::Reflex::NewDelFunctions**)retaddr = &s_funcs;\n'
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
        mod = '::Reflex::' + access.upper()
        if virtual : mod = '::Reflex::VIRTUAL |' + mod
        bases.append( [id,  mod, level] )
        self.getAllBases( id, bases, level+1, access, virtual )
#----------------------------------------------------------------------------------
  def isCopyCtor(self, cid, mid):
    args = self.xref[mid]['subelems']
    if (len(args) == 1 or (len(args) > 1 and 'default' in args[1])) :
      arg0type = args[0]['type']
      while self.xref[arg0type]['elem'] in ( 'ReferenceType', 'CvQualifiedType') :
        arg0type = self.xref[arg0type]['attrs']['type']
      if arg0type == cid: return 1
    return 0
#----------------------------------------------------------------------------------
  def completeClass(self, attrs):
    # Complete class with "instantiated" templated methods or constructors
    # for GCCXML 0.9: add default c'tor, copy c'tor, d'tor if not available.
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
    # GCCXML now (>0.7) takes care by itself of which functions are implicitly defined:
    haveCtor    = 1
    haveCtorCpy = 1
    haveDtor    = 1
    if self.gccxmlvers.find('0.7') == 0:
      haveCtor    = 0
      haveCtorCpy = 0
      haveDtor    = 0
      for m in members :
        if self.xref[m]['elem'] == 'Constructor' :
          haveCtor = 1
          if haveCtorCpy == 0:
            haveCtorCpy = self.isCopyCtor(cid, m)
        elif self.xref[m]['elem'] == 'Destructor' :
          haveDtor = 1
    if haveCtor == 0 :
      id = u'_x%d' % self.x_id.next()
      new_attrs = { 'name':attrs['name'], 'id':id, 'context':cid, 'artificial':'true', 'access':'public' }
      self.xref[id] = {'elem':'Constructor', 'attrs':new_attrs, 'subelems':[] }
      attrs['members'] += u' ' + id
    if haveCtorCpy == 0 :
      ccid = cid + 'c'
      # const cid exists?
      if ccid not in self.xref :
        new_attrs = { 'id':ccid, 'type':cid }
        self.xref[ccid] = {'elem':'ReferenceType', 'attrs':new_attrs }
      # const cid& exists?
      crcid = 0
      for xid in self.xref :
        if self.xref[xid]['elem'] == 'ReferenceType' and self.xref[xid]['attrs']['type'] == ccid :
          crcid = xid
          break
      if crcid == 0:
        crcid = u'_x%d' % self.x_id.next()
        new_attrs = { 'id':crcid, 'type':ccid, 'const':'1' }
        self.xref[crcid] = {'elem':'ReferenceType', 'attrs':new_attrs }

      # build copy ctor
      id = u'_x%d' % self.x_id.next()
      new_attrs = { 'name':attrs['name'], 'id':id, 'context':cid, 'artificial':'true', 'access':'public' }
      arg = { 'type':crcid }
      self.xref[id] = {'elem':'Constructor', 'attrs':new_attrs, 'subelems':[arg] }
      attrs['members'] += u' ' + id
    if haveDtor == 0 :
      id = u'_x%d' % self.x_id.next()
      new_attrs = { 'name':attrs['name'], 'id':id, 'context':cid, 'artificial':'true', 'access':'public' }
      self.xref[id] = {'elem':'Destructor', 'attrs':new_attrs, 'subelems':[] }
      attrs['members'] += u' ' + id      
#---------------------------------------------------------------------------------------
def getContainerId(c):
  if   c[-8:] == 'iterator' : return ('NOCONTAINER','')
  # MSVC9 templated iterators:
  elif c[-14:] == 'iterator<true>'  : return ('NOCONTAINER','')
  elif c[-15:] == 'iterator<false>' : return ('NOCONTAINER','')
  elif c[:10] == 'std::deque'   :            return ('DEQUE','list')
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
  elif c[:11] == 'std::bitset'  :            return ('BITSET','bitset')
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
  begin = cl.find('<')
  if begin == -1 : return []
  end = cl.rfind('>')
  if end == -1 : return []
  args, cnt = [], 0
  for s in string.split(cl[begin+1:end],',') :
    if   cnt == 0 : args.append(s)
    else          : args[-1] += ','+ s
    cnt += s.count('<')+s.count('(')-s.count('>')-s.count(')')
  if len(args) and len(args[-1]) and args[-1][-1] == ' ' :
    args[-1] = args[-1][:-1]
  return args
#---------------------------------------------------------------------------------------
def normalizeClassAllTempl(name)   : return normalizeClass(name,True)
def normalizeClassNoDefTempl(name) : return normalizeClass(name,False)
def normalizeClass(name,alltempl,_useCache=True,_cache={}) :
  if _useCache:
    key = (name,alltempl)
    if _cache.has_key(key):
      return _cache[key]    
    else:
      ret = normalizeClass(name,alltempl,False)
      _cache[key] = ret
      return ret
  names, cnt = [], 0
  # Special cases:
  # a< (0 > 1) >::b
  # a< b::c >
  # a< b::c >::d< e::f >
  for s in string.split(name,'::') :
    if cnt == 0 : names.append(s)
    else        : names[-1] += '::' + s
    cnt += s.count('<')+s.count('(')-s.count('>')-s.count(')')
  normlist = [normalizeFragment(frag,alltempl,_useCache) for frag in names]
  return string.join(normlist, '::')
#--------------------------------------------------------------------------------------
def normalizeFragment(name,alltempl=False,_useCache=True,_cache={}) :
  name = name.strip()
  if _useCache:
    key = (name,alltempl)
    if _cache.has_key(key):
      return _cache[key]    
    else:
      ret = normalizeFragment(name,alltempl,False)
      _cache[key] = ret
      return ret
  if name.find('<') == -1  : 
    nor =  name
    if nor.find('int') == -1: return nor
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
  else : clname = name[:name.find('<')]
  if name.rfind('>') < len(clname) : suffix = ''
  else                             : suffix = name[name.rfind('>')+1:]
  args = getTemplateArgs(name)
  sargs = [normalizeClass(a, alltempl, _useCache=_useCache) for a in args]

  if not alltempl :
    defargs = stldeftab.get (clname)
    if defargs and type(defargs) == type({}):
      args = [normalizeClass(a, True, _useCache=_useCache) for a in args]
      defargs_tup = None
      for i in range (1, len (args)):
        defargs_tup = defargs.get (tuple (args[:i]))
        if defargs_tup:
          lastdiff = i-1
          for j in range (i, len(args)):
            if defargs_tup[j] != args[j]:
              lastdiff = j
          sargs = args[:lastdiff+1]
          break
    elif defargs:
      # select only the template parameters different from default ones
      args = sargs
      sargs = []
      nargs = len(args)
      if len(defargs) < len(args): nargs = len(defargs)
      for i in range(nargs) :  
        if args[i].find(defargs[i]) == -1 : sargs.append(args[i])
    sargs = [normalizeClass(a, alltempl, _useCache=_useCache) for a in sargs]

  nor = clname + '<' + string.join(sargs,',')
  if nor[-1] == '>' : nor += ' >' + suffix
  else              : nor += '>' + suffix
  return nor
#--------------------------------------------------------------------------------------
def clean(a) :
  r = []
  for i in a :
	if i not in r : r.append(i)
  return r
#--------------------------------------------------------------------------------------
# Add implementations of functions declared by ROOT's ClassDef() macro
def ClassDefImplementation(selclasses, self) :
  # test whether Rtypes.h got included:
  haveRtypes = 0
  for file in self.files:
    if self.files[file]['name'].endswith('Rtypes.h') \
           and ( self.files[file]['name'][-9] == '/' or self.files[file]['name'][-9] == '\\' ):
      haveRtypes = 1
      break
  if haveRtypes == 0: return ''
  
  returnValue  = '#ifndef G__DICTIONARY\n' # for RtypesImp.h
  returnValue += '# define G__DICTIONARY\n'
  returnValue += '#endif\n'
  returnValue += '#include "TClass.h"\n'
  returnValue += '#include "TMemberInspector.h"\n'
  returnValue += '#include "RtypesImp.h"\n' # for GenericShowMembers etc
  returnValue += '#include "TIsAProxy.h"\n'
  haveClassDef = 0

  for attrs in selclasses :
    members = attrs.get('members','')
    membersList = members.split()

    listOfMembers = []
    for ml in membersList:
      if ml[1].isdigit() :
        listOfMembers.append(self.xref[ml]['attrs']['name'])

    allbases = []
    self.getAllBases(attrs['id'], allbases)

    # If the class inherits from TObject it MUST use ClassDef; check that:
    derivesFromTObject = 0
    if len(self.TObject_id) :
      if len( filter( lambda b: b[0] == self.TObject_id, allbases ) ) :
        derivesFromTObject = 1

    if "fgIsA" in listOfMembers \
           and "Class" in listOfMembers \
           and "Class_Name" in listOfMembers  \
           and "Class_Version" in listOfMembers  \
           and "Dictionary" in listOfMembers  \
           and "IsA" in listOfMembers  \
           and "ShowMembers" in listOfMembers  \
           and "Streamer" in listOfMembers  \
           and "StreamerNVirtual" in listOfMembers \
           and "DeclFileName" in listOfMembers \
           and "ImplFileLine" in listOfMembers \
           and "ImplFileName" in listOfMembers :

      clname = '::' + attrs['fullname']

      haveClassDef = 1
      extraval = '!RAW!' + str(derivesFromTObject)
      if attrs.has_key('extra') : attrs['extra']['ClassDef'] = extraval
      else                      : attrs['extra'] = {'ClassDef': extraval}
      attrs['extra']['ClassVersion'] = '!RAW!' + clname + '::Class_Version()'
      id = attrs['id']
      template = ''
      namespacelevel = 0
      if clname.find('<') != -1:
        template = 'template<> '
        # specialization of A::B::f() needs namespace A { template<> B<...>::f(){} }
        specclname = attrs['name']
        enclattrs = attrs
        while 'context' in enclattrs:
          if self.xref[enclattrs['id']]['elem'] == 'Namespace' :
            namespname = ''
            if 'fullname' in enclattrs :
              namespname = enclattrs['fullname']
            else :
              namespname = self.genTypeName(enclattrs['id'])
            namespacelevel = namespname.count('::') + 1
            returnValue += 'namespace ' + namespname.replace('::', ' { namespace ')
            returnValue += ' { \n'
            break
          specclname = enclattrs['name'] + '::' + specclname
          enclattrs = self.xref[enclattrs['context']]['attrs']
      else :
        specclname = clname

      returnValue += template + 'TClass* ' + specclname + '::Class() {\n'
      returnValue += '   if (!fgIsA)\n'
      returnValue += '      fgIsA = TClass::GetClass("' + clname[2:] + '");\n'
      returnValue += '   return fgIsA;\n'
      returnValue += '}\n'
      returnValue += template + 'const char * ' + specclname + '::Class_Name() {return "' + clname[2:]  + '";}\n'
      haveNewDel = 0
      if 'GetNewDelFunctions' in listOfMembers:
        haveNewDel = 1
        # need to fwd decl newdel wrapper because ClassDef is before stubs
        returnValue += 'namespace {\n'
        returnValue += '   static void method_newdel' + id + '(void*, void*, const std::vector<void*>&, void*);\n'
        returnValue += '}\n'
      returnValue += template + 'void ' + specclname + '::Dictionary() {}\n'
      returnValue += template + 'const char *' + specclname  + '::ImplFileName() {return "";}\n'

      returnValue += template + 'int ' + specclname + '::ImplFileLine() {return 1;}\n'

      returnValue += template + 'void '+ specclname  +'::ShowMembers(TMemberInspector &R__insp) {\n'
      returnValue += '   TClass *R__cl = ' + clname  + '::IsA();\n'
      returnValue += '   if (R__cl || R__insp.IsA()) { }\n'

      for ml in membersList:
        if ml[1].isdigit() :
          if self.xref[ml]['elem'] == 'Field' :
            mattrs = self.xref[ml]['attrs']
            varname  = mattrs['name']
            tt = self.xref[mattrs['type']]
            te = tt['elem']
            if te == 'PointerType' :
              varname1 = '*' + varname
            elif te == 'ArrayType' :
              t = self.genTypeName(mattrs['type'],colon=True,const=True)
              arraytype = t[t.find('['):]
              varname1 = varname + arraytype
            else :
              varname1 = varname
            # rootcint adds a cast to void* here for the address of the member, as in:
            # returnValue += '   R__insp.Inspect(R__cl, R__parent, "' + varname1 + '", (void*)&' + varname + ');\n'
            # but only for struct-type members. CVS log from 2001:
            #  "add explicit cast to (void*) in call to Inspect() only for object data"
            #  "members not having a ShowMembers() method. Needed on ALPHA to be able to"
            #  "compile G__Thread.cxx."
            returnValue += '   R__insp.Inspect(R__cl, R__insp.GetParent(), "' + varname1 + '", &' + varname + ');\n'
            # if struct: recurse!
            if te in ('Class','Struct') :
              memtypeid = mattrs['type']
              memDerivesFromTObject = (memtypeid == self.TObject_id)
              if not memDerivesFromTObject :
                allmembases = []
                self.getAllBases(memtypeid, allmembases)
                if len( filter( lambda b: b[0] == self.TObject_id, allmembases ) ) :
                  memDerivesFromTObject = 1
              if memDerivesFromTObject :
                returnValue +=  '   R__insp.InspectMember(%s, "%s.");\n' % (varname, varname)
              else :
                # TODO: the "false" parameter signals that it's a non-transient (i.e. a persistent) member.
                # We have the knowledge to properly pass true or false, and we should do that at some point...
                returnValue +=  '   R__insp.InspectMember("%s", (void*)&%s, "%s.", %s);\n' \
                               % (self.genTypeName(memtypeid), varname, varname, "false")
                # tt['attrs']['fullname']

      if 'bases' in attrs :
        for b in attrs['bases'].split() :
          poscol = b.find(':')
          if poscol == -1 : baseid = b
          else            : baseid = b[poscol + 1:]
          basename = self.genTypeName(baseid)
          basemem = self.xref[baseid]['attrs']['members']
          baseMembersList = basemem.split()
          baseHasShowMembers = 0
          for ml in baseMembersList:
            if ml[1].isdigit() :
              if self.xref[ml]['attrs']['name'] == 'ShowMembers' :
                baseHasShowMembers = 1
                break
          # basename = self.xref[baseid]['attrs']['fullname']
          if baseHasShowMembers :
            returnValue +=  '   %s::ShowMembers(R__insp);\n' % basename
          else :
            returnValue +=  '   R__insp.GenericShowMembers("%s", ( ::%s *)(this), false);\n' % (basename, basename)

      returnValue += '}\n'

      returnValue += template + 'void '+ specclname  +'::Streamer(TBuffer &b) {\n   if (b.IsReading()) {\n'
      returnValue += '      b.ReadClassBuffer(' + clname + '::Class(),this);\n'
      returnValue += '   } else {\n'
      returnValue += '      b.WriteClassBuffer(' + clname  + '::Class(),this);\n'
      returnValue += '   }\n'
      returnValue += '}\n'
      returnValue += template + 'TClass* ' + specclname + '::fgIsA = 0;\n'
      returnValue += namespacelevel * '}' + '\n'
    elif derivesFromTObject :
      # no fgIsA etc members but derives from TObject!
      print '--->> genreflex: ERROR: class %s derives from TObject but does not use ClassDef!' % attrs['fullname']
      print '--->>                   You MUST put ClassDef(%s, 1); into the class definition.' % attrs['fullname']

  if haveClassDef == 1 :
    return "} // unnamed namespace\n\n" + returnValue + "\nnamespace {\n"
  return ""

#--------------------------------------------------------------------------------------
# If Class_Version is a member function of the class, use it to set ClassVersion
def Class_VersionImplementation(selclasses, self):
  for attrs in selclasses :
    #if ClassVersion was already set, do not change
    if attrs.has_key('extra') and 'ClassVersion' in attrs['extra']:
        continue
    
    members = attrs.get('members','')
    membersList = members.split()

    hasClass_Version = False
    for ml in membersList:
      memAttrs = self.xref[ml]['attrs']
      if ml[1].isdigit() and attrs.has_key('name') and memAttrs['name'] == "Class_Version":
        if memAttrs.get('access') not in ('protected', 'private') and 'static' in memAttrs:         
            hasClass_Version = True
        else:
            print "--->> genreflex: ERROR: class %s's method Class_Version() must be both 'static' and 'public'." % attrs['fullname']
        break

    if hasClass_Version:
      clname = '::' + attrs['fullname']
      if attrs.has_key('extra'):
        attrs['extra']['ClassVersion'] = '!RAW!' + clname + '::Class_Version()'
      else:
        attrs['extra']={'ClassVersion' : '!RAW!' + clname + '::Class_Version()'}
