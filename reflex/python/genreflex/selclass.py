# Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
#
# Permission to use, copy, modify, and distribute this software for any
# purpose is hereby granted without fee, provided that this copyright and
# permissions notice appear in all copies and derivatives.
#
# This software is provided "as is" without express or implied warranty.

import xml.parsers.expat
import os, sys, string, time, fnmatch

class selClass :
#----------------------------------------------------------------------------------
  def __init__(self, file, parse=0):
    self.file           = file
    self.sel_classes    = []
    self.exc_classes    = []
    self.sel_functions  = []
    self.exc_functions  = []
    self.sel_enums      = []
    self.exc_enums      = []
    self.sel_vars       = []
    self.exc_vars       = []
    self.classes   = self.sel_classes
    self.functions = self.sel_functions
    self.enums     = self.sel_enums
    self.vars      = self.sel_vars
    if parse : self.parse()
#----------------------------------------------------------------------------------
  def parse(self):
    p = xml.parsers.expat.ParserCreate()
    p.StartElementHandler = self.start_element
    p.EndElementHandler = self.end_element
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
    elif name in ('function','operator'):
      self.functions.append({'attrs':attrs})
      if 'name' in attrs :  attrs['name'] = attrs['name'].replace(' ','')
    elif name in ('enum',):
      self.enums.append({'attrs':attrs})
    elif name in ('variable',):
      self.vars.append({'attrs':attrs})
    elif name in ('field',) :
      self.classes[-1]['fields'].append(attrs)
    elif name in ('method',) :
      self.classes[-1]['methods'].append(attrs)
    elif name in ('selection',) :
      self.classes   = self.sel_classes
      self.functions = self.sel_functions
    elif name in ('exclusion',) :
      self.classes   = self.exc_classes
      self.functions = self.exc_functions
    if 'pattern' in attrs :
      attrs['n_pattern'] = self.genNName(attrs['pattern'])
#----------------------------------------------------------------------------------
  def end_element(self, name):
    if name in ('exclusion',) :
      self.classes   = self.sel_classes
      self.functions = self.sel_functions
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
      if 'n_name' in attrs and attrs['n_name'] == clname :  c['used'] = 1; return attrs
      if 'n_pattern' in attrs and matchpattern(clname,attrs['n_pattern']) : return attrs
      if 'file_name' in attrs and attrs['file_name'] == fname : return attrs
      if 'file_pattern' in attrs and matchpattern(fname,attrs['file_pattern']): return attrs
    return None
#----------------------------------------------------------------------------------
  def excclass(self, clname, fname ) :
    for c in self.exc_classes :
      if c['methods'] or c['fields'] : continue
      attrs = c['attrs']
      if 'n_name' in attrs  and attrs['n_name'] == clname : return attrs 
      if 'n_pattern' in attrs and matchpattern(clname, attrs['n_pattern']) : return attrs
      if 'file_name' in attrs and attrs['file_name'] == fname : return attrs
      if 'file_pattern' in attrs and matchpattern(fname,attrs['file_pattern']): return attrs
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
          if 'n_name' in attrs and attrs['n_name'] == clname : return f
          if 'n_pattern' in attrs and matchpattern(clname, attrs['n_pattern']) : return f
    return None
#----------------------------------------------------------------------------------
  def excfield(self, clname, field ) :
    clname = clname.replace(' ','')
    for c in self.exc_classes :
      for f in c['fields'] :
        if 'name' in f and f['name'] == field :
          attrs = c['attrs'] 
          if 'n_name' in attrs and attrs['n_name'] == clname : return f
          if 'n_pattern' in attrs and matchpattern(clname, attrs['n_pattern']) : return f
    return None
#----------------------------------------------------------------------------------
  def matchmethod(self, clname, method ) :
    return self.selmethod(clname, method), self.excmethod(clname,method)
#----------------------------------------------------------------------------------
  def selmethod(self, clname, method ) :
    clname = clname.replace(' ','')
    for c in self.sel_classes :
      for m in c['methods'] :
        if ('name' in m and m['name'] == method ) or ('pattern' in m and matchpattern(method, m['pattern'])) :
          attrs = c['attrs']
          if 'n_name' in attrs and attrs['n_name'] == clname : return m
          if 'n_pattern' in attrs and matchpattern(clname, attrs['n_pattern']) : return m
    return None
#----------------------------------------------------------------------------------
  def excmethod(self, clname, method ) :
    clname = clname.replace(' ','')
    for c in self.exc_classes :
      for m in c['methods'] :
        if ('name' in m and m['name'] == method ) or ('pattern' in m and matchpattern(method, m['pattern'])) :
          attrs = c['attrs']
          if 'n_name' in attrs and attrs['n_name'] == clname : return m
          if 'n_pattern' in attrs and matchpattern(clname, attrs['n_pattern']) : return m
    return None
#----------------------------------------------------------------------------------
  def selfunction(self, funcname ) :
    for f in self.sel_functions :
      attrs = f['attrs']
      if 'name' in attrs and attrs['name'] == funcname :  return attrs
      if 'pattern' in attrs and matchpattern(funcname,attrs['pattern']) : return attrs
    return None
#----------------------------------------------------------------------------------
  def excfunction(self, funcname ) :
    for f in self.exc_functions :
      attrs = f['attrs']
      if 'name' in attrs  and attrs['name'] == funcname : return attrs 
      if 'pattern' in attrs and matchpattern(funcname, attrs['pattern']) : return attrs
    return None
#----------------------------------------------------------------------------------
  def selenum(self, enumname ) :
    for enum in self.sel_enums :
      attrs = enum['attrs']
      if 'name' in attrs and attrs['name'] == enumname :  return attrs
      if 'pattern' in attrs and matchpattern(enumname,attrs['pattern']) : return attrs
    return None
#----------------------------------------------------------------------------------
  def excenum(self, enumname ) :
    for enum in self.exc_enums :
      attrs = enum['attrs']
      if 'name' in attrs  and attrs['name'] == enumname : return attrs 
      if 'pattern' in attrs and matchpattern(enumname, attrs['pattern']) : return attrs
    return None
#----------------------------------------------------------------------------------
  def selvariable(self, varname ) :
    for var in self.sel_vars :
      attrs = var['attrs']
      if 'name' in attrs and attrs['name'] == varname :  return attrs
      if 'pattern' in attrs and matchpattern(varname,attrs['pattern']) : return attrs
    return None
#----------------------------------------------------------------------------------
  def excvariable(self, varname ) :
    for var in self.exc_vars :
      attrs = var['attrs']
      if 'name' in attrs  and attrs['name'] == varname : return attrs 
      if 'pattern' in attrs and matchpattern(varname, attrs['pattern']) : return attrs
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
def matchpattern( name, pattern ) :
  return fnmatch.fnmatch(name.replace('*','#'),pattern.replace('\*','#'))
