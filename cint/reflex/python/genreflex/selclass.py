# Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
#
# Permission to use, copy, modify, and distribute this software for any
# purpose is hereby granted without fee, provided that this copyright and
# permissions notice appear in all copies and derivatives.
#
# This software is provided "as is" without express or implied warranty.

import xml.parsers.expat
import os, sys, string, time, fnmatch
import re

class selClass :
#----------------------------------------------------------------------------------
  def __init__(self, file, parse=0):
    self.file             = file
    self.sel_classes      = []
    self.exc_classes      = []
    self.sel_functions    = []
    self.exc_functions    = []
    self.sel_enums        = []
    self.exc_enums        = []
    self.sel_vars         = []
    self.exc_vars         = []
    self.io_read_rules    = {}
    self.io_readraw_rules = {}
    self.current_io_rule  = None
    self.classes   = self.sel_classes
    self.functions = self.sel_functions
    self.enums     = self.sel_enums
    self.vars      = self.sel_vars
    self.ver_re    = re.compile('^\d+-$|^-\d+$|^\d+$|^(\d+)-(\d+)$') # ie. it matches: 1-,-1,1,1-2
    if parse : self.parse()
#----------------------------------------------------------------------------------
  def parse(self):
    p = xml.parsers.expat.ParserCreate()
    p.StartElementHandler = self.start_element
    p.EndElementHandler = self.end_element
    p.CharacterDataHandler = self.char_data
    f = open(self.file)
    # Replace any occurence of <>& in the attribute values by the xml parameter
    rxml, nxml = f.read(), ''
    q1,q2 = 0,0
    for c in rxml :
      if   (q1 or q2) and c == '<' : nxml += '&lt;'
      elif (q1 or q2) and c == '>' : nxml += '&gt;'
      # elif (q1 or q2) and c == '&' : nxml += '&amp;'
      else                         : nxml += c
      if c == '"' : q1 = not q1
      if c == "'" : q2 = not q2
    try : p.Parse(nxml)
    except xml.parsers.expat.ExpatError, e :
      print '--->> genreflex: ERROR: parsing selection file ',self.file
      print '--->> genreflex: ERROR: Error is:', e
      raise 
    f.close()
#----------------------------------------------------------------------------------
  def genNName(self, name ):
    n_name = string.join(name.split())         
    for e in [ ['long long unsigned int', 'unsigned long long'],
               ['long long int',          'long long'],
               ['unsigned short int',     'unsigned short'],
               ['short unsigned int',     'unsigned short'],
               ['short int',              'short'],
               ['long unsigned int',      'unsigned long'],
               ['unsigned long int',      'unsigned long'],
               ['long int',               'long'],
               ['std::string',            'std::basic_string<char>']] :
      n_name = n_name.replace(e[0],e[1])
    n_name = n_name.replace(' ','')
    return n_name
#----------------------------------------------------------------------------------
  def start_element(self, name, attrs):
    if name in ('class','struct'):
      self.classes.append({'attrs':attrs, 'fields':[], 'methods':[]})
      if 'name' in attrs : attrs['n_name'] = self.genNName(attrs['name'])
    elif name in ('function'):
      self.functions.append({'attrs':attrs})
      if 'name' in attrs :  attrs['name'] = attrs['name'].replace(' ','')
    elif name in ('operator'):
      self.functions.append({'attrs':attrs})
      if 'name' in attrs :
        attrs['name'] = attrs['name'].replace(' ','')
        if attrs['name'][0].isalpha():
          attrs['name'] = 'operator ' + attrs['name']
        else :
          attrs['name'] = 'operator' + attrs['name']
    elif name in ('enum',):
      self.enums.append({'attrs':attrs})
    elif name in ('variable',):
      self.vars.append({'attrs':attrs})
    elif name in ('field',) :
      self.classes[-1]['fields'].append(attrs)
    elif name in ('method',) :
      self.classes[-1]['methods'].append(attrs)
    elif name in ('ioread', 'ioreadraw', 'read', 'readraw'):
      self.current_io_rule = {'attrs': attrs, 'code': '' }
    elif name in ('selection',) :
      self.classes   = self.sel_classes
      self.functions = self.sel_functions
      self.vars      = self.sel_vars
      self.enums     = self.sel_enums
    elif name in ('exclusion',) :
      self.classes   = self.exc_classes
      self.functions = self.exc_functions
      self.vars      = self.exc_vars
      self.enums     = self.exc_enums
    if 'pattern' in attrs :
      attrs['n_pattern'] = self.genNName(attrs['pattern'])
#----------------------------------------------------------------------------------
  def end_element(self, name):
    if name in ('exclusion',) :
      self.classes   = self.sel_classes
      self.functions = self.sel_functions
      self.vars      = self.sel_vars
      self.enums     = self.sel_enums

    #------------------------------------------------------------------------------
    # Processing io rules
    #------------------------------------------------------------------------------
    elif name == 'ioread' or name =='ioreadraw' or name == 'read' or name =='readraw':
      if not self.isRuleValid( self.current_io_rule ):
        print '--->> genreflex: WARNING: The IO rule has been omited'
        self.current_io_rule = None
        return

      className = self.current_io_rule['attrs']['targetClass']

      #----------------------------------------------------------------------------
      # Handle read rule
      #----------------------------------------------------------------------------
      if name == 'ioread' or name == 'read':
        if not self.io_read_rules.has_key( className ):
          self.io_read_rules[className] = []
        self.io_read_rules[className].append( self.current_io_rule )
        self.current_io_rule = None

      #----------------------------------------------------------------------------
      # Handle readraw rule
      #----------------------------------------------------------------------------
      elif name == 'ioreadraw' or name == 'readraw':
        source = self.current_io_rule['attrs']['source'].split(',')
        if len(source) > 1:
          print '--->> genreflex: WARNING: IO rule for class:', className,
          print '- multiple sources speciffied for readraw rule!'
          return
        if not self.io_readraw_rules.has_key( className ):
          self.io_readraw_rules[className] = []
        self.io_readraw_rules[className].append( self.current_io_rule )
        self.current_io_rule = None

      self.current_io_rule = None
#----------------------------------------------------------------------------------
  def isRuleValid(self, rule):

    #------------------------------------------------------------------------------
    # Checks if we have all necessary tags
    #------------------------------------------------------------------------------
    attrs = self.current_io_rule['attrs']
    if not attrs.has_key( 'targetClass' ):
      print '--->> genreflex: WARNING: You always have to specify the targetClass when specyfying an IO rule'
      return False

    className = attrs['targetClass'].strip()
    warning = '--->> genreflex: WARNING: IO rule for class ' + className

    if not attrs.has_key( 'sourceClass' ):
        print warning, '- sourceClass attribute is missing'
        return False

    if not attrs.has_key( 'version' ) and not attrs.has_key( 'checksum' ):
      print warning, '- You need to specify either version or checksum'
      return False

    #------------------------------------------------------------------------------
    # Check if the checksums are correct
    #------------------------------------------------------------------------------
    if attrs.has_key( 'checksum' ):
      chk = attrs['checksum']
      if chk[0] != '[' or chk[-1] != ']':
        print warning, '- a comma separated list of ints enclosed in square brackets expected',
        print 'as a value of checksum parameter'
        return False

      lst = [item.strip() for item in chk[1:-1].split(',')]
      if len( lst ) == 0:
        print warning, 'the checksum list is empty'
        return False

      for chk in lst:
        try:
          if ( chk != "*" ):
             i = int( chk )
        except:
          print warning, chk, 'is not a valid value of checksum parameter - an integer expected'
          return False

    #------------------------------------------------------------------------------
    # Check if the versions are correct
    #------------------------------------------------------------------------------
    if attrs.has_key( 'version' ):
      ver = attrs['version']
      if ver[0] != '[' or ver[-1] != ']':
        print warning, '- a comma separated list of version specifiers enclosed in square',
        print 'brackets expected as a value of version parameter'
        return False

      lst = [item.strip() for item in ver[1:-1].split(',')]
      if len( lst ) == 0:
        print warning, 'the version list is empty'
        return False

      for v in lst:
        if ( v != "*" ):
          matchObj = self.ver_re.match( v )
          if not matchObj:
            print warning, '-', v, 'is not a valid value of version parameter'
            return False
          else:
            rng = matchObj.groups()
            if rng[0] and rng[1]:
              b, e = int(rng[0]), int(rng[1])
              if b >= e:
                print warning, '-', v, 'is not a valid version range'
                return False

    #------------------------------------------------------------------------------
    # Check if we deal with renameing rule
    #------------------------------------------------------------------------------
    if len( attrs ) == 3 or (len( attrs ) == 4 and attrs.has_key( 'version' ) and attrs.has_key( 'checksum' )):
      return True

    #------------------------------------------------------------------------------
    # Check if we have other parameters specified correctly
    #------------------------------------------------------------------------------
    #  source and target are optional paramater.
    #  for k in ['target', 'source' ]:
    #  if not attrs.has_key(k):
    #    print warning, '- Required attribute is missing:', k
    #    return False

    if attrs.has_key( 'embed' ):
      if attrs['embed'] != 'true' and attrs['embed'] != 'false':
        print warning, '- true or false expected as a value of embed parameter'
        return False

    #------------------------------------------------------------------------------
    # Check if the include list is not empty
    #------------------------------------------------------------------------------
    if attrs.has_key( 'include' ):
      if len( attrs['include'] ) == 0:
        print warning, 'empty include list specified'
        return False

    return True

#----------------------------------------------------------------------------------
  def char_data(self, data):
    if self.current_io_rule:
      self.current_io_rule['code'] += data
#----------------------------------------------------------------------------------
  def matchclassTD(self, clname, fname, sltor) :
    clname = clname.replace(' ','')
    for s in sltor :
      if 'name' in s['attrs'] : s['attrs']['n_name'] = self.genNName(s['attrs']['name'])
    return self.selclass(clname, fname, sltor), self.excclass(clname, fname)
#----------------------------------------------------------------------------------
  def matchclass(self, clname, fname ) :
    clname = clname.replace(' ','')
    return self.selclass(clname, fname, self.sel_classes), self.excclass(clname, fname)
#----------------------------------------------------------------------------------
  def selclass(self, clname, fname, sltor ) :
    for c in sltor :
      attrs = c['attrs']
      if 'n_name' in attrs and attrs['n_name'] == clname \
            or 'n_pattern' in attrs and self.matchpattern(clname,attrs['n_pattern']) \
            or 'file_name' in attrs and attrs['file_name'] == fname \
            or 'file_pattern' in attrs and self.matchpattern(fname,attrs['file_pattern']) :
        c['used'] = 1
        if c.has_key('fields'):
          attrs['fields'] = c['fields']
        return attrs
    return None
#----------------------------------------------------------------------------------
  def excclass(self, clname, fname ) :
    for c in self.exc_classes :
      if c['methods'] or c['fields'] : continue
      attrs = c['attrs']
      if 'n_name' in attrs  and attrs['n_name'] == clname : return attrs 
      if 'n_pattern' in attrs and self.matchpattern(clname, attrs['n_pattern']) : return attrs
      if 'file_name' in attrs and attrs['file_name'] == fname : return attrs
      if 'file_pattern' in attrs and self.matchpattern(fname,attrs['file_pattern']): return attrs
    return None
#----------------------------------------------------------------------------------
  def matchfield(self, clname, field ) :
    return self.selfield(clname, field), self.excfield(clname, field)
#----------------------------------------------------------------------------------
  def selfield(self, clname, field ) :
    clname = clname.replace(' ','')
    for c in self.sel_classes :
      for f in c['fields'] :
        if 'name' in f and f['name'] == field :
          attrs = c['attrs'] 
          if 'n_name' in attrs and attrs['n_name'] == clname \
                or 'n_pattern' in attrs and self.matchpattern(clname, attrs['n_pattern']) :
            return f
    return None
#----------------------------------------------------------------------------------
  def excfield(self, clname, field ) :
    clname = clname.replace(' ','')
    for c in self.exc_classes :
      for f in c['fields'] :
        if 'name' in f and f['name'] == field :
          attrs = c['attrs'] 
          if 'n_name' in attrs and attrs['n_name'] == clname : return f
          if 'n_pattern' in attrs and self.matchpattern(clname, attrs['n_pattern']) : return f
    return None
#----------------------------------------------------------------------------------
# unused
  def matchmethod(self, clname, method, arguments ) :
    return self.selmethod(clname, method, arguments), self.excmethod(clname,method, arguments)
#----------------------------------------------------------------------------------
# unused - only excmethod is used
  def selmethod(self, clname, method, arguments ) :
    clname = clname.replace(' ','')
    for c in self.sel_classes :
      for m in c['methods'] :
        if ('name' in m and m['name'] == method ) \
               or ('pattern' in m and self.matchpattern(method, m['pattern'])) \
               or ('proto_name' in m and m['proto_name'] == method + '(' + arguments + ')' ) \
               or ('proto_pattern' in m and self.matchpattern(method + '(' + arguments + ')', m['proto_pattern'])) :
          attrs = c['attrs']
          if 'n_name' in attrs and attrs['n_name'] == clname : return m
          if 'n_pattern' in attrs and self.matchpattern(clname, attrs['n_pattern']) : return m
    return None
#----------------------------------------------------------------------------------
  def excmethod(self, clname, method, demangled ) :
    clname = clname.replace(' ','')
    for c in self.exc_classes :
      for m in c['methods'] :
        if ('name' in m and m['name'] == method ) \
           or ('pattern' in m and self.matchpattern(method, m['pattern'])) \
           or ('proto_name' in m and m['proto_name'] == demangled ) \
           or('proto_pattern' in m and self.matchpattern(demangled, m['proto_pattern'])) :
          attrs = c['attrs']
          if 'n_name' in attrs and attrs['n_name'] == clname : return m
          if 'n_pattern' in attrs and self.matchpattern(clname, attrs['n_pattern']) : return m
    return None
#----------------------------------------------------------------------------------
  def selfunction(self, funcname, demangled ) :
    for f in self.sel_functions :
      attrs = f['attrs']
      if ('name' in attrs and attrs['name'] == funcname ) \
             or ('pattern' in attrs and self.matchpattern(funcname, attrs['pattern'])) \
             or ('proto_name' in attrs and attrs['proto_name'] == demangled ) \
             or ('proto_pattern' in attrs and self.matchpattern(demangled, attrs['proto_pattern'])) :
        return attrs
    return None
#----------------------------------------------------------------------------------
  def excfunction(self, funcname, demangled ) :
    for f in self.exc_functions :
      attrs = f['attrs']
      if ('name' in attrs and attrs['name'] == funcname ) \
             or ('pattern' in attrs and self.matchpattern(funcname, attrs['pattern'])) \
             or ('proto_name' in attrs and attrs['proto_name'] == demangled ) \
             or ('proto_pattern' in attrs and self.matchpattern(demangled, attrs['proto_pattern'])) :
        return attrs
    return None
#----------------------------------------------------------------------------------
  def selenum(self, enumname ) :
    for enum in self.sel_enums :
      attrs = enum['attrs']
      if 'name' in attrs and attrs['name'] == enumname :  return attrs
      if 'pattern' in attrs and self.matchpattern(enumname,attrs['pattern']) : return attrs
    return None
#----------------------------------------------------------------------------------
  def excenum(self, enumname ) :
    for enum in self.exc_enums :
      attrs = enum['attrs']
      if 'name' in attrs  and attrs['name'] == enumname : return attrs 
      if 'pattern' in attrs and self.matchpattern(enumname, attrs['pattern']) : return attrs
    return None
#----------------------------------------------------------------------------------
  def selvariable(self, varname ) :
    for var in self.sel_vars :
      attrs = var['attrs']
      if 'name' in attrs and attrs['name'] == varname :  return attrs
      if 'pattern' in attrs and self.matchpattern(varname,attrs['pattern']) : return attrs
    return None
#----------------------------------------------------------------------------------
  def excvariable(self, varname ) :
    for var in self.exc_vars :
      attrs = var['attrs']
      if 'name' in attrs  and attrs['name'] == varname : return attrs 
      if 'pattern' in attrs and self.matchpattern(varname, attrs['pattern']) : return attrs
    return None
#----------------------------------------------------------------------------------
  def reportUnusedClasses(self) :
    warnings = 0
    for c in self.sel_classes :
      if 'name' in c['attrs'] and 'used' not in c :
         print '--->> genreflex: WARNING: Class %s in selection file %s not generated.' % (c['attrs']['name'] , self.file )
         warnings += 1
    return warnings
#-----------------------------------------------------------------------------------
  def matchpattern(self, name, pattern ) :
    return fnmatch.fnmatch(name.replace('*','#'),pattern.replace('\*','#'))
