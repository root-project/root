# Pretty printers for gdb
# \author: Stephan Hageboeck, CERN
# These pretty printers will make ROOT objects more readable when printed in gdb.
# If the pretty-printed output is not sufficient, one can always use "print /r <object>"
# for raw printing.
#
# When a debug build is used, they will be installed next to the ROOT libraries.
# gdb will load them automatically if the auto-load-safe-path is set to ROOT's library directory.
# For this, one has to add `add-auto-load-safe-path <ROOT lib dir>` to .gdbinit
#
# If loaded successfully, typing `info pretty-printer` at the gdb prompt should list the
# printers registered at the end of this file.

import gdb
import gdb.printing

class TObjectPrinter(object):
   "Print TObjects"

   def __init__(self, val):
      self.__val = val
      
   def children(self):
      yield "fUniqueID", self.__val['fUniqueID']
      yield "fBits", self.__val['fBits']

   def to_string(self):
      return self.__val.dynamic_type.name



class TNamedPrinter(object):
   "Print TNamed"

   def __init__(self, val):
      self.__val = val
   
   def children(self):
      yield "<TObject>", self.__val[self.__val.type.fields()[0]]

   def to_string(self):
      return "(" + str(self.__val['fName']) + ", " + str(self.__val['fTitle']) + ")"



class TStringPrinter(object):
   "Print TStrings"

   def __init__(self, val):
      self.__val = val
      typeAndAddr = "(*(TString*)"+str(val.address)+")"
      query = typeAndAddr + ".fRep.fShort.fSize & TString::kShortMask"
      self.isLong = bool(gdb.parse_and_eval(query))
      
   def display_hint(self):
      return 'string'

   def to_string(self):
      theStr = self.__val['fRep']['fLong']['fData'] if self.isLong else self.__val['fRep']['fShort']['fData']
      return theStr.string()




def build_pretty_printer():
   pp = gdb.printing.RegexpCollectionPrettyPrinter("libCore.so")
   pp.add_printer('TObject', '^TObject$', TObjectPrinter)
   pp.add_printer('TNamed', '^TNamed$', TNamedPrinter)
   pp.add_printer('TString', '^TString$', TStringPrinter)  

   return pp


gdb.printing.register_pretty_printer(gdb.current_objfile(),
    build_pretty_printer())
