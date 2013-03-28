# Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
#
# Permission to use, copy, modify, and distribute this software for any
# purpose is hereby granted without fee, provided that this copyright and
# permissions notice appear in all copies and derivatives.
#
# This software is provided "as is" without express or implied warranty.

import xml.parsers.expat
import string

classes = []
namespaces = []
xref = {}
last_id = ''

def genName(id) :
  if id[-1] == u'c' :
    return  'const ' + genName(id[:-1])
  elem  = xref[id][0]
  attrs = xref[id][1]
  if elem == 'PointerType' :
    return genName(attrs[u'type'])+'*'
  elif elem == 'ReferenceType' :
    return genName(attrs[u'type'])+'&'
  elif elem == 'FunctionType' :
    s = genName(attrs[u'returns']) + '(*)('
    children = xref[id][2]
    for a in children :
      s += genArgument(a)
      if a != children[-1] : s += ', '
    s += ')'
    return s
  elif elem == 'ArrayType' :
    return genName(attrs[u'type'])+'['+attrs[u'max']+']'
  else :
    return attrs[u'name']
def genField(attrs, children):
  return '%s %s;' % (genName(attrs[u'type']), attrs[u'name'] )
def genVariable(attrs, children):
  return 'static %s %s;' % (genName(attrs[u'type']), attrs[u'name'] )
def genArgument(attrs):
  if u'name' in attrs : 
    return '%s %s' % (genName(attrs[u'type']), attrs[u'name'] )
  else :
    return '%s ' % (genName(attrs[u'type']))  
def genMethod(attrs, children):
  s = ''
  if u'virtual' in attrs : s += 'virtual '
  if u'static' in attrs : s += 'static '
  s += '%s %s(' % (genName(attrs[u'returns']), attrs[u'name'])
  for a in children : 
    s += genArgument(a)
    if a != children[-1] : s += ', '
  s += ')'
  if u'const' in attrs : s += ' const'
  if u'pure_virtual' in attrs : s += ' = 0'
  s += ';'
  return s
def genConstructor(attrs, children):
  s = '%s(' % (attrs[u'name'])
  for a in children : 
    s += genArgument(a)
    if a != children[-1] : s += ', '
  s += ');'
  return s
def genOperatorMethod(attrs, children):
  s = '%s operator %s(' % ( genName(attrs[u'returns']), attrs[u'name'])
  for a in children : 
    s += genArgument(a)
    if a != children[-1] : s += ', '
  s += ')'
  if u'const' in attrs : s += ' const'
  s += ';'
  return s
def genDestructor(attrs, children):
  return '~%s();' % (attrs[u'name'])
def genConverter(attrs, children):
  return 'operator %s();' % (attrs[u'returns'])
def genEnumValue(attrs):
  return '%s = %s' % (attrs[u'name'], attrs[u'init'])
def genEnumeration(attrs, children):
  s = 'enum %s { ' % (attrs[u'name'])
  for a in children :
    s += genEnumValue(a)
    if a != children[-1] : s += ', '
  s += '};'
  return s
def genClass(attrs, children ):
  s = 'class %s ' % (attrs[u'name'])
  if u'bases' in attrs :
    bases = string.split(attrs[u'bases'])
    if bases :
      s += ': '
      for b in bases :
        if b[0] == '_' : 
          s += genName(b)
        elif b[0:9] == 'protected:' : 
          s += 'protected '+ genName(b[10:])
        if b != bases[-1] : s += ', '
  s += ' {\n'
  if u'members' in attrs :
    members   = string.split(attrs[u'members'])
    for m in members:
      funcname = 'gen'+xref[m][0]
      if funcname in globals() :
        s += '  ' + apply(globals()[funcname],(xref[m][1], xref[m][2])) + '\n'
      else :
        print 'Function '+funcname+' not found'
  s += '};'
  return s
def genTypedef(attrs, children):
  return 'typedef %s %s;' % ( genName(attrs[u'type']), attrs[u'name'] )
  
def start_element(name, attrs):
  global last_id
  if u'id' in attrs :
    xref[attrs[u'id']] = (name, attrs, [])
    last_id = attrs[u'id']
  if name in ('EnumValue','Argument') :
    xref[last_id][2].append(attrs)
  elif name == 'Class' :
    classes.append(attrs)
  elif name == 'Namespace' :
    namespaces.append(attrs)
def end_element(name):
  pass
    
p = xml.parsers.expat.ParserCreate()
p.StartElementHandler = start_element
p.EndElementHandler = end_element

fp = open('..\\data\\MCParticle.xml')
p.ParseFile(fp)

for c in classes :
  print genClass( c, [] ) 
