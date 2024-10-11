## \file
## \ingroup tutorial_v7
## \notebook
##
##
## This PyRoot 7 example demonstrates how to use RNTuple in combination with 
## ROOT 6 features like `RDataframe` and its visualizations.
## It ingests climate data (on the cloud) and creates a model with fields 
## like "AverageTemperature", "CityName", "Day", among others.
## Then it uses `RDataframe` to process and filter the climate data to get 
## the average temperature per city and per season. 
## Then it does the same but for the average temperature per city over the years 
## between 1993-2002, and 2003-2013. 
## Finally, the tutorial visualizes the processed data through histograms
## on two canvases in a web browser.
## 
##
## TODO(jblomer): 
##                Re-enable once issues are fixed (\macro_image (rcanvas_js))
##                Setting colors at the TObjectDrawable doesn't display 
##                correctly on the histograms
##                but only on the legends. 
##                Invoking `RCanvas::Draw( TObjectDrawable );` has 
##                an internal ambiguous function call `GetDrawable(...)`.
##                On the Python side, `RCanvas.Draw["TObjectDrawable"](...)` 
##                fails to instantiate.
##                
##
##
## NOTE: Until C++ runtime modules are »universally« used, we explicitly have 
##       to load the ntuple library. 
##       Otherwise,
##       triggering the autoloading option for the template types
##       would require an exhaustive enumeration
##       of »all« template instances in the "LinkDef" file.
##
## \warning The RNTuple classes are experimental at this point.
##          Functionality, interface, and data format are still 
##          inclined to changes.
##          Do not use it for real data! 
##          This macro requires certain requirements. During ROOT setup, 
##          configure the following flags:
##          `-DCMAKE_CXX_STANDARD=17 -Droot7=ON -Dwebgui=ON`
##
##
## \copy_right
## Aclaratory note for data rights.
##    Climate data is downloadable at the following URL:
##    https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data
##    The original data set is from http://berkeleyearth.org/archive/data/
##    License CC BY-NC-SA 4.0
##
## \macro_image
## \macro_output
## \macro_code
##
## \date 2021-02-26
## \author John Yoon
## \translator P. P.


import re
import ctypes

import ROOT
import cppyy


# standard library
from ROOT import std
from ROOT.std import (
                       make_shared,
                       unique_ptr,
                       )


# standard c library
libc = ctypes.CDLL("libc.so.6")
# sscanf
sscanf = libc.sscanf



# root 7
# experimental classes
from ROOT.Experimental import (
                               RDrawable,
                               RCanvas,
                               RColor,
                               RHistDrawable,
                               RNTuple,
                               RNTupleDS,
                               RNTupleModel,
                               RNTupleWriteOptions,
                               RNTupleWriter,
                               # RNTupleOptions,
                               # RNTupleOptions,
                               # RRawFile,
                               )

#
from ROOT import RDataFrame
from ROOT.RDF import (
                       TH1DModel,
                       )
# Not to use:
#from ROOT.RDF.Experimental import (
#                                    FromRNTuple,
#                                    )
# Instead:
from ROOT.RDF import Experimental as RDF_Experimental 
FromRNTuple = RDF_Experimental.FromRNTuple



# classes
from ROOT import (
                   TH1D,
                   TLegend,
                   TSystem,
                   TCanvas,
                   TFile,
                   TH1F,
                   TGraph,
                   TLatex,
                   TCanvas,
                   TPaveLabel,
                   TPavesText,
                   TPaveText,
                   TText,
                   TArrow,
                   TWbox,
                   TPad,
                   TBox,
                   TPad,
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
                   Float_t,
                   Int_t,
                   nullptr,
)
#
from ctypes import (
                     c_double,
                     c_int,
                     c_uint,
                     c_float,
                     c_char_p,
                     c_void_p,
                     create_string_buffer,
                     #
                     byref,
                     POINTER,
                     cast,
                     )
#
c_string = create_string_buffer


# utils
def to_c( ls ):
   return ( c_double * len( ls ) )( * ls )
def printf( string, * args ):
   print( string % args )
def sprintf( buffer, string, * args ):
   buffer = string % args 
   return buffer


# constants
from ROOT import (
                   kBlack,
                   kOrange,
                   kViolet,
                   kSpring,
                   kBlue,
                   kRed,
                   kGreen,
                   )

# globals
from ROOT import (
                   gStyle,
                   gPad,
                   gRandom,
                   gBenchmark,
                   gROOT,
                   gInterpreter,
                   )


# c-integration
from ROOT.gInterpreter import (
                                ProcessLine,
                                Declare,
                                )

Clock = std.chrono.high_resolution_clock




# Alternative definition to RNTupleWriter.Recreate for Python side.
Declare("""
using namespace ROOT::Experimental;
using namespace std; 

unique_ptr<RNTupleWriter>
RNTupleWriter_Recreate_Py ( unique_ptr<RNTupleModel> &model,
                               string_view ntupleName,
                               string_view storage,
                               const RNTupleWriteOptions &options ){

   return RNTupleWriter::Recreate( move(model), ntupleName, storage, options);
};

""")
RNTupleWriter_Recreate_Py = ROOT.RNTupleWriter_Recreate_Py



# Alternative definition for sscanf for Python side.
def sscanf_Py( string, str_format, * args ):

   # Set-up libc.sscanf
   libc.sscanf.argtypes
   #
   # To have something like this :
   # libc.sscanf.argtypes = [
   #                          ctypes.c_char_p,        
   #                          ctypes.c_char_p,        
   #                          ctypes.POINTER(ctypes.c_uint),        
   #                          ctypes.POINTER(ctypes.c_uint),        
   #                          ctypes.POINTER(ctypes.c_uint),        
   #                          ctypes.POINTER(ctypes.c_float),        
   #                          ctypes.POINTER(ctypes.c_float),
   # 
   #                         ]
   # Only available formats.
   # "d" int
   # "f" float
   # "s" str
   #
   type_correspondence = {
                           "%s" : c_char_p ,
                           "%d" : c_int    ,
                           "%u" : c_uint   ,
                           "%f" : c_float  ,
                           } 

   #
   sscanf.argtypes = [ ]
   types = re.findall( r"(%\w)", str_format )
   for typ in types :
      sscanf.argtypes.append( 
                             type_correspondence[ typ ]
                             )

   #
   addressof = cppyy.addressof
   # Concept :
   #    cast( addressof( _ ), POINTER( __ctypes. )
   #    byref( _.contents ) 
   # Prototype :
   # pointers = map( 
   #                 #lambda  _obsj, _typs
   #                 lambda _,  __ : cast( addressof( _ ), POINTER( __ ) ),
   #                 #lambda _objs,  _typs : print( _objs, POINTER( _typs ) ),
   #                 #lambda _objs,  _typs : print( _objs,  _typs  ),
   #                 args,
   #                 sscanf.argtypes,
   #                 )
   # pointers = list( pointers )
   # references = map( 
   #                  lambda _ptrs : byref( _ptrs.contents ),
   #                  pointers,
   #                  )
   # references = list( references )
   # 
   # TODO
   # We do not use the above yet since the "cast" process
   # producess extra floaing numbers at scanning float types;
   # like: 
   #        1.234 (original data)  ->  1.23456789 (after cast process)
   # A more shorter but limited version is implemented instead. 
   references = list( map( byref, args ) )
   
   # This should be in general.
   libc.sscanf.restype = c_int

   return sscanf ( string.encode( 'utf-8' )     ,
                   str_format.encode( 'utf-8' ) ,
                   * references                 ,
                   )
          


# Original code:
ProcessLine("""
// Helper function to handle histogram pointer ownership.
std::shared_ptr<TH1D> GetDrawableHist(ROOT::RDF::RResultPtr<TH1D> &h) {
   auto result = std::shared_ptr<TH1D>(static_cast<TH1D *>(h.GetPtr()->Clone()));
   result->SetDirectory(nullptr);
   return result;
}
""")
GetDrawableHist = ROOT.GetDrawableHist
#
# TODO:
# Python Prototype: 
#  # Helper function to handle histogram pointer ownership.
#  #std.shared_ptr["TH1D"] 
#  def GetDrawableHist( h : ROOT.RDF.RResultPtr["TH1D"] ) :
#     
#     result = std.shared_ptr["TH1D"]( 
#                 BindObject( addressof( h.GetPtr().Clone() ), TH1D )
#                 )
#     result = h.GetPtr().Clone() # TH1D
#     result.SetDirectory(nullptr)
#     return result
#     


# Introducing the wrapper function for the
# `ROOT::Experimental::RCanvas::Draw`
# method function.
#                  To solve this:
#                    canvas.Draw["TObjectDrawable"]( fallHist   , "L" )
#                    canvas.Draw["TObjectDrawable"]( winterHist , "L" )
#                    canvas.Draw["TObjectDrawable"]( springHist , "L" )
#                    canvas.Draw["TObjectDrawable"]( summerHist , "L" )
#                     
# First. 
# The shell command for compiling the DrawHistogram_wrapper is:
#   you@terminal:~/root/tutorials/v7 $                 \
#                   g++                                \
#                        -shared                       \
#                        -fPIC                         \
#                        -o histogram_wrapper.so       \
#                        histogram_wrapper.cpp         \
#                        `root-config --cflags --libs` \
#                        -fmax-errors=1                
#             
# Now, adapting that inside Python with `subprocess` module.
import subprocess
command = ( " g++                              "                            
            "     -shared                      " 
            "     -fPIC                        " 
            "     -o histogram_wrapper.so      " 
            "     histogram_wrapper.cpp        " 
            "     `root-config --cflags --libs`" 
            "     -fmax-errors=1"               ,               
            )
# Then execute.
try : 
   subprocess.run( command, shell=True )
except :
   raise RuntimeError( "histogram_wrapper.cpp function not well compiled.")
# Then load `histogram_wrapper.cpp` .
cppyy.load_library( "./histogram_wrapper.so" )
cppyy.include     ( "./histogram_wrapper.cpp" )
from cppyy.gbl import (
                        DrawHistogram_wrapper,
                        DrawLegend_wrapper,
                        )
# At last, the wrapper function is ready.

# To this point, all settings have been done for a minimum use of ROOT 7 in
# Python 3. Let's procede with the analysis: write, read and plot.




# Parameters.
kRawDataUrl = \
      "http://root.cern.ch/files/tutorials/GlobalLandTemperaturesByCity.csv"
kNTupleFileName = "GlobalLandTemperaturesByCity.root"
kMaxCharsPerLine = 128 # In ".csv" file of original data.


# Functions.

# void
def Ingest() :

   print( " >>> Converting \n"                   ,
                                 kRawDataUrl     ,
          "\n into \n"                           ,
                                 kNTupleFileName ,
          )
   
   
   t1 = Clock.now()
   
   # Create a unique pointer to an empty data model.
   global model
   model = RNTupleModel.Create()
   # 1.
   # To define the data model, 
   # create fields with a given C++ type and name.  
   # 2.
   # Fields are roughly TTree branches.
   # 3.
   # MakeField returns a shared pointer to a memory location 
   # to fill the ntuple with data.
   #
   global                    \
            fieldYear,       \
            fieldMonth,      \
            fieldDay,        \
            fieldAvgTemp,    \
            fieldTempUncrty, \
            fieldCity,       \
            fieldCountry,    \
            fieldLat,        \
            fieldLong
   #
   fieldYear                = model.MakeField["std::uint32_t"]("Year")
   fieldMonth               = model.MakeField["std::uint32_t"]("Month")
   fieldDay                 = model.MakeField["std::uint32_t"]("Day")
   fieldAvgTemp             = model.MakeField["float"]("AverageTemperature")
   fieldTempUncrty          = model.MakeField["float"]("AverageTemperatureUncertainty")
   #
   # Note : 
   #        No support for:
   #        model.MakeField["std::string"]("City")  
   #        TODO : Create automatization of this in future PyROOT versions > 6.30.02.
   # 
   name_City    = RNTupleModel.NameWithDescription_t("City")
   name_Country = RNTupleModel.NameWithDescription_t("Country")
   # 
   fieldCity                = model.MakeField["std::string"]( name_City )
   fieldCountry             = model.MakeField["std::string"]( name_Country )
   #
   #
   fieldLat                 = model.MakeField["float"]("Latitude")
   fieldLong                = model.MakeField["float"]("Longitude")
   #
   # Creation of ctypes as an intermediate layer before assigning its values.
   c_fieldYear       = c_uint( )
   c_fieldMonth      = c_uint( )
   c_fieldDay        = c_uint( )
   c_fieldAvgTemp    = c_float( )
   c_fieldTempUncrty = c_float( )
   c_fieldCity       = c_string( kMaxCharsPerLine )
   c_fieldCountry    = c_string( kMaxCharsPerLine )
   c_fieldLat        = c_float( )
   c_fieldLong       = c_float( )


   
   # Hand-over the data model to a newly created ntuple 
   # of name "globalTempData", which will be stored in `kNTupleFileName`.
   # In return, get a unique pointer to a 
   # fillable ntuple (first compress the file).
   #
   global options, ntuple
   options = RNTupleWriteOptions()
   options.SetCompression( ROOT.RCompressionSetting.EDefaults.kUseGeneralPurpose )


   # Not to use:
   # ntuple = RNTupleWriter.Recreate(std.move( model ),
   #                                 "GlobalTempData",
   #                                 kNTupleFileName,
   #                                 options,
   #                                 )
   # Instead:
   global ntuple
   ntuple = RNTupleWriter_Recreate_Py( std.move( model )  ,
                                       "GlobalTempData" ,
                                       kNTupleFileName  ,
                                       options          ,
                                       )
   # Or:
   # ntuple = RNTupleWriter_Recreate_Py( model,
   #                                     "GlobalTempData",
   #                                     kNTupleFileName,
   #                                     options,
   #                                     )

   
   # Get the on-the-cloud data. Link .
   global file
   file = ROOT.Internal.RRawFile.Create( kRawDataUrl )
   # Set its buffer: temporary line.
   global record
   record = std.string( )

   
   # Reading the first line: titles.
   file.Readln( record ) 
   print( " Columns :" )
   print( "          ", record )


   #
   # Loop over the data on the cloud.
   #
   nRecords = 0
   nSkipped = 0
   i = 0
   while ( file.Readln( record ) ) :
      i += 1
      #
      # Reading the first lines only.
      # if i > 10 * 1000000: break # Short but still large.
      # if i > 10 * 100000: break  # Good for tests.
      if i > 1 * 100000: break  # Good for tests.
      # if i > 10 : break # Quick tests.


      #
      # Checking quality of data.
      #
      if ( record.length( ) >= kMaxCharsPerLine ) :
         raise RuntimeError( "Record too long: " + record, 
               f"Only kMaxCharsPerLine: {kMaxCharsPerLine} permitted." )
      

      #
      # Scanning.
      #
      # Parse lines like this:
      #    1743-11-01,6.068,1.7369999999999999,Århus,Denmark,57.05N,10.33E
      # into:
      #     fieldYear
      #     fieldMonth
      #     fieldDay
      #     fieldAvgTemp
      #     fieldTempUncrty
      #     fieldCity
      #     fieldCountry
      #     fieldLat
      #     fieldLong
      #
      #
      # Changing format. Replace "," by " ".
      # Before:   1743-11-01,6.068,1.7369999999999999,Århus,Denmark,57.05N,10.33E
      # After:    1743-11-01 6.068 1.7369999999999999 Århus Denmark 57.05N 10.33E
      std.replace[""]( record.begin(), record.end(), ",", " ")
      #
      # Note:
      #     - We use "std.replace" c-template since "record" is a reference
      #       to a c-object.
      #     - We leave the template parameters with nothing, so c++
      #       will infere the types.
      #     - std.replace is very limited for operations but is faster 
      #       than python str.replace. The replacement has to be done
      #       a character for a character, in this case. 
      #       A complete instantiation of the std.replace template 
      #       should be:
      #   std_replace_tmplt = std.replace["std::string::iterator, char"] 
      #   std_replace_tmplt ( record.begin(), record.end(), ',', ' ')
     
      # Define the scan pattern to be matched.
      global pattern
      pattern = '%u-%u-%u %f %f %s %s %fN %fE'
      # for lines with the pattern:
      #      '1743-11-01 6.068 1.7369999999999999 Århus Denmark 57.05N 10.33E'
      #
      # If you need it:
      # pattern = "%u-%u-%u %f %f %*s %*s %*fN %*fE"
      # The "*" in the "pattern" means ignore this field, so it won't scan it.


      #
      # Scan: match process.
      #
      # 1. Search pattern.
      #
      nFields = sscanf_Py( record.c_str()    ,
                           pattern           ,

                           c_fieldYear       ,
                           c_fieldMonth      ,
                           c_fieldDay        ,
                           c_fieldAvgTemp    ,
                           c_fieldTempUncrty ,
                           c_fieldCountry    ,
                           c_fieldCity       ,
                           c_fieldLat        ,
                           c_fieldLong       ,
                           )
      #
      # 2. Loading the »c_fields« into the smart pointers.
      #
      fieldYear       . get( ) [ 0 ] = c_fieldYear       . value
      fieldMonth      . get( ) [ 0 ] = c_fieldMonth      . value
      fieldDay        . get( ) [ 0 ] = c_fieldDay        . value
      fieldAvgTemp    . get( ) [ 0 ] = c_fieldAvgTemp    . value
      fieldTempUncrty . get( ) [ 0 ] = c_fieldTempUncrty . value
      fieldCountry    . assign(        c_fieldCountry    . value )
      fieldCity       . assign(        c_fieldCity       . value )
      fieldLat        . get( ) [ 0 ] = c_fieldLat        . value
      fieldLong       . get( ) [ 0 ] = c_fieldLong       . value
      #
      # Important Notes :
      #        - Note the difference in assigning strings.
      #        - Up to now, 
      #          we could have been assign 
      #          "fieldCity" and "fieldCountry",
      #          but we want to add one more step to the analysis
      #          to make it »interesting«:
      #             We will add a restriction for when and how
      #             to fill those fields( City and Country ). 
      #             This is possible because the output of 
      #             sscanf_Py, a.k.a. libc.so.6.sscanf,
      #             returns how many fields have been scanned 
      #             and filled.
      #             Thus, we can save the number of fields 
      #             just read as "nField".
      #             Then, our restriction will be defined in terms 
      #             of "nField":
      #             >>>   nField = sscanf_Py( ... )
      #             >>>   columns = 9 
      #             >>>   if nField =! columns: continue
      #             which basically »marks« those lines with a 
      #             deficient data by not labeling them.
      #             So, bad data will not be count.
       

                           


      #
      # Restriction.
      # All fields haven't been count in the current line. 
      if (nFields != 9) :
         nSkipped += 1  
         continue

      #
      # Fill exceptions after passing restriction point.
      #
      fieldCountry . assign( str( c_fieldCountry ) )
      fieldCity    . assign( str( c_fieldCity    ) )


      # 
      # Filling fields into the RNTuple-object.
      #
      ntuple.Fill( )

      #
      # Status report.      
      #
      nRecords += 1
      if ( nRecords % 1000000 == 0):
         print(f"  ... converted " , nRecords , " records")
      
   

   #
   # Number of Lines skipped and processed. 
   #
   print(nSkipped , " records skipped")
   print(nRecords , " records processed")
   

   #
   # Display the total time needed to process the data.
   #
   t2 = Clock.now()
   print(
          "\n"                                                              ,
          "Processing Time: "                                               ,
          std.chrono.duration_cast["std::chrono::seconds"](t2 - t1).count() ,
          "  seconds.\n"                                                    ,
            
           )
   


# Note:
#      Every data result that we want to obtain is declared first, 
#      and it is only upon that declaration that they are actually 
#      well used. 
#      This stems from motivations relating to efficiency and optimization. 
#      Take that into account.
#
# void
def Analyze() :

   #
   # Create a RDataframe by wrapping around NTuple.
   # Not to use:
   # df = ROOT.RDF.Experimental.FromRNTuple("GlobalTempData", kNTupleFileName)
   #             Error:
   #                  Fatal: nread == nbytes violated at line 1083 of 
   # `/home/pp/Projects/root/root_src2/tree/ntuple/v7/src/RMiniFile.cxx'
   #                  aborting  
   # Instead:
   ProcessLine( f"""
   auto df = ROOT::RDF::Experimental::\
             FromRNTuple(
                          "GlobalTempData", 
                          \"{ kNTupleFileName }\"
                          );
   """)
   #
   global df
   df = ROOT.df
   df.Display().Print()
   

   
   # Declare the minimum and maximum temperature from the dataset.
   min_value = df.Min("AverageTemperature").GetValue()
   max_value = df.Max("AverageTemperature").GetValue()
   


   # Defining 
   # functions that filter by each season from formatted-date-data like "1944-12-01."
   #
   # Note : Lambda-functions and any other kind of Python functions are not 
   #        fully supported in ROOT v6.32.
   #
   # fnWinter = lambda month : month == 12 or month == 1 or month == 2
   # fnSpring = lambda month : month == 3 or month == 4 or month == 5
   # fnSummer = lambda month : month == 6 or month == 7 or month == 8
   # fnFall   = lambda month : month == 9 or month == 10 or month == 11
   #
   # Using C++ lambda functions instead will not work either:
   #
   # ProcessLine("""
   #    auto fnWinter = [](int month) { return month == 12 || month == 1 || month == 2; };
   #    auto fnSpring = [](int month) { return month == 3 || month == 4 || month == 5; };
   #    auto fnSummer = [](int month) { return month == 6 || month == 7 || month == 8; };
   #    auto fnFall   = [](int month) { return month == 9 || month == 10 || month == 11; };
   # """)
   # fnWinter = ROOT.fnWinter
   # fnSpring = ROOT.fnSpring
   # fnSummer = ROOT.fnSummer
   # fnFall = ROOT.fnFall
   #
   #
   # Whichever python function(a lambda or a pure function) isn't supported yet.
   # 
   # Instead, we can define a python function and convert it to string using
   # the "inspect" module of Python as follows:

   # # 1. Some Python functions with logic-operators, they should return a boolean type.
   # #
   def fnWinter (month) : return month == 12 or month == 1 or month == 2 
   def fnSpring (month) : return month == 3 or month == 4 or month == 5 
   def fnSummer (month) : return month == 6 or month == 7 or month == 8 
   def fnFall   (month) : return month == 9 or month == 10 or month == 11 

   # # 2. Use `inspect` and `re` modules to convert a python function into 
   # #    a string-type. This converts also the logical-operator into C-syntax.
   # #
   # #    TODO : Improve. This is a prototype, but it is a start.
   # #
   # #
   # # 2.1 Defininig a helper function `to_string`.
   # #
   import inspect
   import re 
   def to_string( py_func ):
      """ to string in c-syntax """

      source = inspect.getsource( py_func ) 

      # take care of ; at the end of line.
      source = source.removesuffix(";\n") 

      # Changing logic operators to c-syntax.
      source = re.sub( " or " , " || ", source )
      source = re.sub( " and ", " && ", source )

      if not py_func.__name__ == "<lambda>" and \
         pyfunc.__class__.__name__ == "function":

         #source = inspect.getsource( py_func ) 
         # def f( ... return ....
         body =  source.split( "return" )[ -1 ].strip()

      elif py_func.__name__ == "<lambda>" : 

         #source = inspect.getsource( py_func ) 
         # f = lambda ... : ....
         body =  source.split( "lambda" )[ -1 ].split(":")[-1].strip()
      
      else :
         raise TypeError( "py_func should be a python function or a lambda function" )
     
      # Adding ( ) as a single unit of logic expresion.
      return f"( {body} )"

   # # 3. It should work as:
   # #
   # # fnWinter = lambda month : month == 12 or month == 1 or month == 2
   # # fnWinter_str = to_string( fnWinter ) 
   # # dfWinter = df.Filter( fnWinter_str, "Month")
   # # 
   # # 
   # # 4. Or as an alternative, 
   # # You could just use the C-syntax directly in the `.Filter` method:
   # #
   # # dfWinter = df.Filter("Month == 9 || Month == 10 || Month == 11", "Month" )

   
   #
   # Create a RDataFrame per season.
   #
   # # Note: `.Filter` method is not yet fully implemented in PyRoot < v6.32.04 .
   # #       The `.Filter` method lacks of python functions type.
   # # Not to use:
   # #             dfWinter = df.Filter(fnWinter, ["Month"])
   # #             dfSpring = df.Filter(fnSpring, "Month")
   # #             dfSummer = df.Filter(fnSummer, "Month")
   # #             dfFall = df.Filter(fnFall, "Month")
   # # 
   # # Instead:
   global dfWinter, dfSpring, dfSummer, dfFall
   dfWinter = df.Filter(" Month == 12 || Month == 1 || Month == 2", "Month" )
   dfSpring = df.Filter(" Month == 3 || Month == 4 || Month == 5   ", "Month")
   dfSummer = df.Filter(" Month == 6 || Month == 7 || Month == 8   ", "Month")
   dfFall   = df.Filter(" Month == 9 || Month == 10 || Month == 11 ", "Month")
   #
   # # Notice the C-syntax for the logic operators `||` instead of `or`.
   # # The `to_string` function does this automatically, but take into account
   # #   that the variable names, should coincide(lowercase and uppercase) exactly
   # #   with the names of the data stored in the RDataFrame object. Otherwise,
   # #   it won't work around.
   
   

   
   # Get the count for each season.
   winterCount  = dfWinter.Count().GetValue()
   springCount  = dfSpring.Count().GetValue()
   summerCount  = dfSummer.Count().GetValue()
   fallCount    =   dfFall.Count().GetValue()
   
   # Functions to filter for the time period between 2003-2013, and 1993-2002.
   fn1993_to_2002 = lambda Year : Year >= 1993 and Year <= 2002
   fn2003_to_2013 = lambda Year : Year >= 2003 and Year <= 2013
   # Using the the python alternative.
   fn1993_to_2002_str = to_string( fn1993_to_2002 ) 
   fn2003_to_2013_str = to_string( fn2003_to_2013 ) 
   
   # Create a RDataFrame for decades 1993_to_2002 & 2003_to_2013.
   df1993_to_2002 = df.Filter(fn1993_to_2002_str, "Year")
   df2003_to_2013 = df.Filter(fn2003_to_2013_str, "Year")
   # Or simply use:
   # df1993_to_2002 = df.Filter( " Year >= 1993 && Year <= 2002" , "Year")
   # df2003_to_2013 = df.Filter( " Year >= 2003 && Year <= 2013" , "Year")
   
   
   # Get the count for each decade.
   decade_1993_to_2002_Count = df1993_to_2002.Count().GetValue()
   decade_2003_to_2013_Count = df2003_to_2013.Count().GetValue()
   
   
   # Configure histograms for each season.
   global fallHistResultPtr
   fallHistResultPtr = \
         dfFall.Histo1D(
                        TH1DModel( "Fall Average Temp"             ,
                                   "Average Temperature by Season" ,
                                   100                             ,
                                   -40                             ,
                                   40                              ,
                                   )         ,
                        "AverageTemperature" ,
                        )
   

   winterHistResultPtr = \
         dfWinter.Histo1D(
                           TH1DModel( "Winter Average Temp"           ,
                                      "Average Temperature by Season" ,
                                      100                             ,
                                      -40                             ,
                                      40                              ,
                                      )         ,
                           "AverageTemperature" ,
                           )
   

   springHistResultPtr = \
         dfSpring.Histo1D(
                           TH1DModel( "Spring Average Temp"           ,
                                      "Average Temperature by Season" ,
                                      100                             ,
                                      -40                             ,
                                      40                              ,
                                      )         ,
                           "AverageTemperature" ,
                           )
   

   summerHistResultPtr = \
         dfSummer.Histo1D(
                           TH1DModel( "Summer Average Temp"           ,
                                      "Average Temperature by Season" ,
                                      100                             ,
                                      -40                             ,
                                      40                              ,
                                      )         ,
                           "AverageTemperature" ,
                           )
   

   
   # Configure histograms for each decade.
   hist_1993_to_2002_ResultPtr = \
         df1993_to_2002.Histo1D(
                                 TH1DModel( "1993_to_2002 Average Temp" ,
                                            "Average Temperature: " +
                                               "1993_to_2002 vs. "  +
                                               "2003_to_2013"           ,
                                            100                         ,
                                            -40                         ,
                                            40                          ,
                                            )         ,
                                 "AverageTemperature" ,
                                 )

   hist_2003_to_2013_ResultPtr = \
         df2003_to_2013.Histo1D(
                                 TH1DModel( "2003_to_2013 Average Temp" ,
                                            "Average Temperature:" +
                                               " 1993_to_2002 vs." +
                                               " 2003_to_2013"          ,
                                            100                         ,
                                            -40                         ,
                                            40                          ,
                                            )         ,
                                 "AverageTemperature" ,
                                 )
   

   
   # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   # Plot process. 
   #


   
   # Display the minimum and maximum temperature values.
   #
   print(                                                            )
   print( "The Minimum temperature is: " , min_value                 )
   print( "The Maximum temperature is: " , max_value                 )
   
   # Display the count for each season.
   #
   print(                                                            )
   print( "The count for Winter: "       , winterCount               )
   print( "The count for Spring: "       , springCount               )
   print( "The count for Summer: "       , summerCount               )
   print( "The count for Fall: "         , fallCount                 )
   
   # Display the count for each decade.
   #
   print(                                                            )
   print( "The count for 1993_to_2002: " , decade_1993_to_2002_Count )
   print( "The count for 2003_to_2013: " , decade_2003_to_2013_Count )
   
   
   # Transform histogram in order to address ROOT 7 v 6 version compatibility.
   global fallHist, winterHist, springHist, summerHist
   fallHist   = GetDrawableHist( fallHistResultPtr   ) # shared_ptr<TH1D>
   winterHist = GetDrawableHist( winterHistResultPtr ) # shared_ptr<TH1D>
   springHist = GetDrawableHist( springHistResultPtr ) # shared_ptr<TH1D>
   summerHist = GetDrawableHist( summerHistResultPtr ) # shared_ptr<TH1D>

   
   # Set an orange histogram for fall.
   fallHist.SetLineColor( kOrange )
   #fallHist.SetFillColor( kOrange ) # Fill down the curve.
   fallHist.SetLineWidth( 6 )

   # Set a blue histogram for winter.
   winterHist.SetLineColor( kBlue )
   #winterHist.SetFillColor( kBlue )
   winterHist.SetLineWidth( 6 )

   # Set a green histogram for spring.
   springHist.SetLineColor( kGreen )
   #springHist.SetFillColor( kGreen )
   springHist.SetLineWidth( 6 )

   # Set a red histogram for summer.
   summerHist.SetLineColor( kRed )
   #summerHist.SetFillColor( kRed )
   summerHist.SetLineWidth( 6 )
 
   

   # Transform histogram in order to address ROOT 7 v 6 version compatibility.
   hist_1993_to_2002 = GetDrawableHist( hist_1993_to_2002_ResultPtr )  # TH1D
   hist_2003_to_2013 = GetDrawableHist( hist_2003_to_2013_ResultPtr )  # TH1D
                                                                   
   # Set a violet histogram for 1993_to_2002.                      
   hist_1993_to_2002.SetLineColor( kViolet )
   hist_1993_to_2002.SetLineWidth( 6 )

   # Set a spring-green histogram for 2003_to_2013.
   hist_2003_to_2013.SetLineColor( kSpring )
   hist_2003_to_2013.SetLineWidth( 6 )

   
   # Create a canvas to display histograms 
   # -- with average temperature by season information --.
   global canvas
   canvas = RCanvas.Create("Average Temperature by Season") # RCanvas
   #
   # Draw on the canvas the histograms.
   # Error:
   # canvas.Draw["TObjectDrawable"]( fallHist   , "L" )
   # canvas.Draw["TObjectDrawable"]( winterHist , "L" )
   # canvas.Draw["TObjectDrawable"]( springHist , "L" )
   # canvas.Draw["TObjectDrawable"]( summerHist , "L" )
   # Instead:
   DrawHistogram_wrapper( canvas, fallHist   , "L" ) # TObjectDrawable *
   DrawHistogram_wrapper( canvas, winterHist , "L" ) # TObjectDrawable *
   DrawHistogram_wrapper( canvas, springHist , "L" ) # TObjectDrawable *
   DrawHistogram_wrapper( canvas, summerHist , "L" ) # TObjectDrawable *
   # 
   # About the error:
   # The above is a short expression for...
   # canvas.Draw( std.move( RDrawable( fallHist.GetName() )  ) )
   # which still generates the error:
   """
            ==== SOURCE BEGIN ====
       __attribute__((used)) extern "C" void __dtor_115(void* obj, unsigned long nary, int withFree)
       {
          if (withFree) {
             if (!nary) {
                delete (std::shared_ptr<ROOT::Experimental::RDrawable>*) obj;
             }
             else {
                delete[] (std::shared_ptr<ROOT::Experimental::RDrawable>*) obj;
             }
          }
          else {
             typedef std::shared_ptr<ROOT::Experimental::RDrawable> Nm;
       if (!nary) {
                ((Nm*)obj)->~Nm();
             }
             else {
                do {
                   (((Nm*)obj)+(--nary))->~Nm();
                } while (nary);
             }
          }
       }
       
         ==== SOURCE END ====
       Error in <TClingCallFunc::ExecDestructor>: Called with no wrapper, not implemented!
       
   """

   

   # Create a legend for the seasons canvas.
   global legend 
   legend = std.shared_ptr["TLegend"]( 
                                      TLegend( 0.15,
                                               0.65,
                                               0.53,
                                               0.85,
                                               )
                                      )
   #
   legend.AddEntry ( fallHist   , "fall"   , "l" )
   legend.AddEntry ( winterHist , "winter" , "l" )
   legend.AddEntry ( springHist , "spring" , "l" )
   legend.AddEntry ( summerHist , "summer" , "l" )
   
   # Error:
   # canvas.Draw["TObjectDrawable"](legend, "L")
   # Instead:
   DrawLegend_wrapper( canvas, legend, "L" )
   canvas.Show()

  
   # Create a canvas to display histograms for average 
   # temperature for 1993_to_2002 & 2003_to_2013.
   global canvas2
   canvas2 = RCanvas.Create( "Average Temperature:"
                             " 1993_to_2002 vs."
                             " 2003_to_2013"
                             ) # RCanvas
   
   # Error:
   # canvas2.Draw["TObjectDrawable"](hist_1993_to_2002, "L")
   # canvas2.Draw["TObjectDrawable"](hist_2003_to_2013, "L")
   # Instead:
   DrawHistogram_wrapper( canvas2, hist_1993_to_2002, "L" )
   DrawHistogram_wrapper( canvas2, hist_2003_to_2013, "L" )

   
   # Create a legend for the two decades.
   # Error:
   # legend2 = std.make_shared["TLegend"](0.1, 0.7, 0.48, 0.9)
   # Instead:
   global legend2
   legend2 = std.shared_ptr["TLegend"]( 
                                       TLegend( 0.1  ,
                                                0.7  ,
                                                0.48 ,
                                                0.9  ,
                                                ) 
                                       )
   #
   legend2.AddEntry ( hist_1993_to_2002 , "1993_to_2002" , "l" )
   legend2.AddEntry ( hist_2003_to_2013 , "2003_to_2013" , "l" )
   #
   # Error:
   # canvas2.Draw["TObjectDrawable"](legend2, "L")
   # Instead:
   DrawLegend_wrapper( canvas2, legend2, "L" )
   canvas2.Show()
   

# void
def global_temperatures() :

   # If NOT zero (the file does NOT already exist), then Ingest.
   if ( ROOT.gSystem.AccessPathName( kNTupleFileName ) != 0 ) :
      Ingest()
   Analyze()
   


if __name__ == "__main__":
   global_temperatures()

