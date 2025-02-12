## \file
## \ingroup tutorial_tree
## \notebook -nodraw
##
##
## Read data (like CERN staff information) from an ascii file 
## and create a root file from it with a Tree.
##
##
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import re
import ctypes
from array import array
import ROOT


# standard library
from ROOT import std
from ROOT.std import (
                       make_shared,
                       unique_ptr,
                       #
                       fopen,
                       fgets,
                       sscanf,
                       fclose,
                       )



# classes
from ROOT import (
                   TTree,
                   TFile,
                   TCanvas,
                   TH1F,
                   TGraph,
                   TLatex,
)

# maths
from ROOT import (
                   sin,
                   cos,
                   sqrt,
)

# types
from ROOT import (
                   Double_t,
                   Bool_t,
                   Float_t,
                   UInt_t,
                   Int_t,
                   Char_t,
                   nullptr,
)

# ctypes
from ctypes import (
                     c_double,
                     c_int,
                     c_uint,
                     c_char,
                     create_string_buffer,
)

#utils
def to_c( ls ):
   return (c_double * len(ls) )( * ls )
def printf(string, *args):
   print( string % args, end="")
def sprintf(buffer, string, *args):
   buffer = string % args 
   return buffer

# constants
from ROOT import (
                   kBlue,
                   kRed,
                   kGreen,
#
                   kWritePermission,
                   kFileExists,
)

# globals
from ROOT import (
                   gSystem,
                   gStyle,
                   gPad,
                   gRandom,
                   gBenchmark,
                   gROOT,
)



# TFile
def cernbuild(getFile : Int_t = 0, print_opt : Int_t = 1) :
   
   #
   # Types for the Tree structure.
   #
   types = { 
      "Category" : c_int(),           # Int_t()
      "Flag"     : c_uint(),          # UInt_t()
      "Age"      : c_int(),           # Int_t()
      "Service"  : c_int(),           # Int_t()
      "Children" : c_int(),           # Int_t()
      "Grade"    : c_int(),           # Int_t()
      "Step"     : c_int(),           # Int_t()
      "Hrweek"   : c_int(),           # Int_t()
      "Cost"     : c_int(),           # Int_t()
      "Division" : ( c_char * 4 )(),  # Char_t [4]
      "Nation"   : ( c_char * 3 )(),  # Char_t [3]
   }

   Category = c_int()          # Int_t()
   Flag     = c_uint()         # UInt_t()
   Age      = c_int()          # Int_t()
   Service  = c_int()          # Int_t()
   Children = c_int()          # Int_t()
   Grade    = c_int()          # Int_t()
   Step     = c_int()          # Int_t()
   Hrweek   = c_int()          # Int_t()
   Cost     = c_int()          # Int_t()
   Division = ( c_char * 4 )() # Char_t [4]
   Nation   = ( c_char * 3 )() # Char_t [3]
   
  
   # Loading them in the local scope
   for name in types.keys() :
      globals()[ name ] = types[ name ]
      #locals()[ name ] = types[ name ] # unnexpected behaviour.
   #
   # Note :
   #        Seems labouriously at first defining the branch
   #        types like this, but it'll help us to maintain 
   #        them in one place and use them everywhere. 
   #        Particularly at emulating "sdt.sscanf" behaviour.
   

   #
   # The input file "cernstaff.dat" is a copy of the CERN staff data base
   # from 1988.
   #
   filename = "cernstaff.root"           # TString
   tut_dir  = gROOT.GetTutorialDir()     # TString
   Dir      = tut_dir.Data() + "/tree/"
   Dir      = Dir.replace("/./", "/")
   #
   # global fp
   # fp = fopen( "%scernstaff.dat"%Dir, "r") # FILE *
   global fp
   fp = open( "%scernstaff.dat" % Dir, "r") 
   

   #
   global hfile
   hfile = TFile() # * # 0
   #
   # If the argument "getFile" is 1, return the file "cernstaff.root".
   if (getFile)  :
      #
      # If the file does not exist, it will be created.
      if not (gSystem.AccessPathName(Dir + "cernstaff.root", kFileExists)):
         # In $ROOTSYS/tutorials/tree.
         hfile = TFile.Open(Dir + "cernstaff.root")  
         if (hfile)  :
            return hfile
            
         
      # Otherwise try "$PWD/cernstaff.root".
      if not (gSystem.AccessPathName( "cernstaff.root", kFileExists))  :
         # In current directory.
         hfile = TFile.Open("cernstaff.root")  
         if (hfile)  :
            return hfile
            
         
   #      
   # No "cernstaff.root" file found. Must generate it!
   # Generate cernstaff.root in $ROOTSYS/tutorials/tree 
   # if we have write access permissions.
   if (gSystem.AccessPathName( ".", kWritePermission ) )  :
      printf("You must run the script in a directory \
             with write access permissions.\n")
      return 0
      

   print( " >>> " )
   print( "Creating the file ", filename, "\n" )
   #
   hfile = TFile.Open(filename, "RECREATE")
   #
   global tree
   tree = TTree("T", "CERN 1988 staff data")  # TTree
   #
   # globals()[ "Category"] # Alternatives for minizing "type construction for Branches".
   tree.Branch ( "Category" , Category,  "Category/I" )
   tree.Branch ( "Flag"     , Flag,      "Flag/i"     )
   tree.Branch ( "Age"      , Age,       "Age/I"      )
   tree.Branch ( "Service"  , Service,   "Service/I"  )
   tree.Branch ( "Children" , Children,  "Children/I" )
   tree.Branch ( "Grade"    , Grade,     "Grade/I"    )
   tree.Branch ( "Step"     , Step,      "Step/I"     )
   tree.Branch ( "Hrweek"   , Hrweek,    "Hrweek/I"   )
   tree.Branch ( "Cost"     , Cost,      "Cost/I"     )
   tree.Branch ( "Division" , Division,  "Division/C" )
   tree.Branch ( "Nation"   , Nation,    "Nation/C"   )

   #
   #line = [ char() for _ in range(80) ]
   #global line
   #line = create_string_buffer( 80 )
   #
   #while ( fgets(line, 80, fp) )  : # error
   #   #
   for line in fp:
      # error :
      #         Not pythonized yet sscanf.  
      #sscanf( line[0],
      #        "%d %d %d %d %d %d %d  %d %d %s %s",
      #        Category,
      #        Flag,
      #        Age,
      #        Service,
      #        Children,
      #        Grade,
      #        Step,
      #        Hrweek,
      #        Cost,
      #        Division,
      #        Nation,
      #        )

      # It is not trivil the inference of
      # a string into a python_type.#
      # This is very short version.
      #
      # def convert_to_py_type( string ):   # Choose a better name.
      def convert_to_convertible( string ):
         if   string.isalpha( ):   return string.encode( "utf-8" ) 
         elif string.isdigit( ):   return [ int( string   )   ]   
         elif string.isnumeric( ): return [ float( string )   ]
         else :                    return [ 0                 ]

      #
      # Here comes the large part. Without using std.sscanf. Only Python.
      # 
      (
       Category.value,
       Flag.value,
       Age.value,
       Service.value,
       Children.value,
       Grade.value,
       Step.value,
       Hrweek.value,
       Cost.value,
       Division.value,
       Nation.value,

      # Alternatives : do not work.
      # map( lambda x: print( x[0] , x[1] ), \
      # map( lambda x: x[0].value.__class__( x[1] ) , \
      # map( lambda x: x[0].__class__( convert_to_py_type( x[1] ) ) , \
      #
      #
      #            ... >>> convert_to_py_type first, then ctypes will handle the C-convertion.      
      ) = \
      map( lambda x: x[0].__class__( * convert_to_convertible( x[1] ) ).value , \
           zip( types.values(), \

              re.match( 
                       r"\s*"      + 
                       r"(\d+)\s+" +     # ... Category
                       r"(\d+)\s+" +     # ... Flag
                       r"(\d+)\s+" +     # ... Age
                       r"(\d+)\s+" +     # ... Service
                       r"(\d+)\s+" +     # ... Children
                       r"(\d+)\s+" +     # ... Grade
                       r"(\d+)\s+" +     # ... Step
                       r"(\d+)\s+" +     # ... Hrweek
                       r"(\d+)\s+" +     # ... Cost
                       r"(\w+)\s+" +     # ... Division
                       r"(\w+)\s*" +     # ... Nation
                       r"$",

                    line,

                    ).groups() 
              # 
              ) # end-zip
          # 
          ) # end-map 

     
      # Note:
      #       This method is large, but it works.
      #       We have to define thrice.
      #       It has to be only once.
      #       
      #       TODO: Shorten the use of types.
      
      #
      tree.Fill()
      
   #
   if (print_opt)  :
      tree.Print()
      

   #
   tree.Write()
   hfile.Close()
   
   #
   #fclose( fp ) # error
   fp.close()


   #
   hfile.Close()
   gROOT.Remove( hfile )
   del hfile

   #
   # We come here when the script is executed outside 
   # $ROOTSYS/tutorials/tree .
   if (getFile)  :
      hfile = TFile.Open( filename ) 
      return hfile
      
   return 0
   


if __name__ == "__main__":
   cernbuild()
   #
   #cernbuild(1)
