from ROOT import *

from ROOT import kBlack, kRed, kGreen, kBlue, kMagenta, kCyan, kOrange, kAzure, kViolet, kPink, kYellow, kSpring, kGray, kTeal, kWhite, kPlus, kDot

import os, re

colour_counter = 0

def getColour( reset = False ):
  '''
  cycle through colours

  Keyword arguments:
  reset  -- start over again
  '''
  colours = [kBlack,
             kRed,
             kGreen,
             kBlue,
             kMagenta,
             kCyan,
             kOrange,
             kAzure,
             kViolet,
             kPink,
             kYellow,
             kSpring,
             kGray,
             kTeal]
  global colour_counter
  if reset:
    colour_counter = 0
  try:
    retval = colours[colour_counter]
    colour_counter+=1
  except IndexError:
    colour_counter = 1
    return colours[0]
  else:
    return retval

def find_size(alg,level):
   '''
   determine the size of the compressed file for an algorithm and compression level setting

   Keyword arguments:
   alg   -- algorithm number
   level -- compression level
   '''
   filename = 'size.'+str(alg)+'.'+str(level)+'.root'
   if os.path.isfile(filename):
      return os.stat(filename).st_size
   raise ValueError('file not found: ' + filename)
find_size.__name__ = "compressed size"

def find_memread(alg,level):
   '''
   parse massif profile for reading a compressed file. peak memory usage w/o overhead is sought.

   Keyword arguments:
   alg   -- algorithm number
   level -- compression level
   '''
   filename = 'massif-read.'+str(alg)+'.'+str(level)+'.out'
   if os.path.isfile(filename):
      f = open(filename)
      lines = f.readlines()
      try:
         ind = lines.index('heap_tree=peak\n')
         toread = ind-3
         peakmem = int(lines[toread].split('=')[1])
         return peakmem
      except ValueError:
         raise
   raise ValueError('file not found: ' + filename)
find_memread.__name__ = "memory reading (peak)"

def find_realwrite(alg,level):
   '''
   read wall clock time from log file of writing a compressed file.

   Keyword arguments:
   alg   -- algorithm number
   level -- compression level
   '''
   filename = "realtime-write."+str(alg)+'.'+str(level)+".log"
   if os.path.isfile(filename):
      f = open(filename)
      lines = f.readlines()
      try:
        res = float(lines[0].split(' ')[-1])
        return res
      except ValueError:
        raise ValueError('could not parse content of ' + filename)
   raise ValueError('file not found: ' + filename)
find_realwrite.__name__ = "realtime writing"


def find_realread(alg,level):
   '''
   read wall clock time from log file of reading a compressed file.

   Keyword arguments:
   alg   -- algorithm number
   level -- compression level
   '''
   filename = "realtime-read."+str(alg)+'.'+str(level)+".log"
   if os.path.isfile(filename):
      f = open(filename)
      lines = f.readlines()
      try:
        res = float(lines[0].split(' ')[-1])
        return res
      except ValueError:
        raise ValueError('could not parse content of ' + filename)
   raise ValueError('file not found: ' + filename)
find_realread.__name__ = "realtime reading"


def find_memwrite(alg,level):
   '''
   parse massif profile for writing a compressed file. peak memory usage w/o overhead is sought.

   Keyword arguments:
   alg   -- algorithm number
   level -- compression level
   '''
   filename = 'massif-write.'+str(alg)+'.'+str(level)+'.out'
   if os.path.isfile(filename):
      f = open(filename)
      lines = f.readlines()
      try:
         ind = lines.index('heap_tree=peak\n')
         toread = ind-3
         peakmem = int(lines[toread].split('=')[1])
         return peakmem
      except ValueError:
         raise
   raise ValueError('file not found: ' + filename)
find_memwrite.__name__ = "memory writing (peak)"



def find_callwrite(alg,level):
   '''
   parse callgrind profile for writing a compressed file. cycles of R__zipMultipleAlgorithm and the functions called by it are sought.

   Keyword arguments:
   alg   -- algorithm number
   level -- compression level
   '''
   filename = 'callgrind-write.'+str(alg)+'.'+str(level)+'.out'
   if os.path.isfile(filename):
      f = open(filename)
      lines = f.readlines()
      try:
         ind = -1
         for l in lines:
             if re.match('.*R__zipMultipleAlgorithm.*',l):
                ind = lines.index(l)
         if ind < 0:
             raise ValueError("didn't find R__zipMultipleAlgorithm in "+filename)
         toread = ind+1
         pointer = None
         for word in lines[toread].split(' '):
            if re.match("0x..*",word):
               pointer = word
         if pointer:
           for l in lines:
             if re.match('calls.* '+pointer+'.*',l):
                return int(lines[lines.index(l)+1].split(' ')[-1])
         raise ValueError('callcount not found')
      except ValueError:
         print "failure in find_callwrite, searching ",filename
         raise
   raise ValueError('file not found: ' + filename)
find_callwrite.__name__ = "cycles writing (function)"

def find_callread(alg,level):
   '''
   parse callgrind profile for reading a compressed file. cycles of R__unzip and the functions called by it are sought. The unzipping of the header is not accounted.

   Keyword arguments:
   alg   -- algorithm number
   level -- compression level
   '''
   filename = 'callgrind-read.'+str(alg)+'.'+str(level)+'.out'
   if os.path.isfile(filename):
      f = open(filename)
      lines = f.readlines()
      try:
         ind = -1
         for l in lines:
             if re.match('.*R__unzip.*',l):
              if not re.match('.*R__unzip_header.*',l):
               if not re.match('.*R__unzipLZMA.*',l):
                ind = lines.index(l)
         if ind < 0:
             raise ValueError("didn't find R__unzip in "+filename)
         toread = ind+1
         pointer = None
         for word in lines[toread].split(' '):
            if re.match("0x..*",word):
               pointer = word
         if pointer:
           for l in lines:
             if re.match('calls.* '+pointer+'.*',l):
                return int(lines[lines.index(l)+1].split(' ')[-1])
         raise ValueError('callcount not found')
      except ValueError:
         print "failure in find_callread, searching ",filename
         raise
   raise ValueError('file not found: ' + filename)
find_callread.__name__ = "cycles reading (function)"

import numpy


def runstats(stats = [find_size, find_memread, find_realwrite, find_realread, find_memwrite, find_callwrite, find_callread]):
   maxlevel = 10
   alglookup = { 1: "ZLIB",
              2: "LZMA",
              4: "LZO",
              5: "LZ4",
              6: "Zopfli",
              7: "Brotli" 
            }

   graphmap = {}
   print "          | ",
   for func in stats:
       print "{:35s}".format(func.__name__),
   print
   for alg in alglookup:
      columns = []
      algresult = []
      for level in range(1,maxlevel):
          try:
             vals = []
             for func in stats:
                vals.append(func(alg,level))
                #print 'executed ', func.__name__, ' for alg ', alg, ' at level ', level, '\tobtained ', vals[-1]
          except ValueError as detail:
                print "couldn't determine all inputs for alg " + str(alg) + " at level " + str(level) + "due to ",detail
             #raise
          else:
             algresult.append(vals)
          print "{:7s}-{:2d}| ".format(alglookup[alg],level),
          for n in algresult[-1]:
              print "{:32f} | ".format(float(n)),
          print
          #print 'vals ', vals
          #print 'algresult ', algresult
      from numpy import array
      #print 'algresult ', algresult
      algresults = array(algresult)
      #print 'algresults ', algresults
      for column in range(len(stats)):
         the_column = algresults[:,column]
         #print 'the_column ', the_column
         columns.append(the_column)
      graphmap[alg] = columns
      
   
   #print graphmap
   for xaxis in range(len(stats)-1):
      for yaxis in range(xaxis+1,len(stats)):
         canvas = TCanvas()
         first = True
         xmax = array([ graphmap[alg][xaxis].max() for alg in alglookup ]).max()
         ymax = array([ graphmap[alg][yaxis].max() for alg in alglookup ]).max()
         h = TH1F("h","h",100,0,1.05*xmax)
         #h.GetYaxis().SetRangeUser(0,ymax*1.05)
         for b in range(1,100):
            h.SetBinContent(b,ymax)
         h.SetLineColor(kWhite)
         h.SetMarkerColor(kWhite)
         h.SetMarkerStyle(kDot)
         h.GetXaxis().SetTitle(stats[xaxis].__name__)
         h.GetYaxis().SetTitle(stats[yaxis].__name__)
         h.Draw()
         graphs = []
         print "drawing frame"
         for alg in alglookup:
             x = graphmap[alg][xaxis]
             y = graphmap[alg][yaxis]
             #points = TGraph(len(x),x,y)
             points = TGraph(len(x))
             graphs.append(points)
             for i in range(len(x)):
                points.SetPoint(i,x[i],y[i])
             points.GetXaxis().SetTitle(stats[xaxis].__name__)
             points.GetYaxis().SetTitle(stats[yaxis].__name__)
             points.SetTitle(alglookup[alg])
             points.SetName(str(alglookup[alg])+str(xaxis)+str(yaxis))
             points.SetFillColor(kWhite)
             points.SetLineColor(getColour(first))
             points.SetMarkerColor(points.GetLineColor())
                    # for uncompressed
                    #points.SetLineColor(kWhite)
                    #points.Draw("Psame")
             points.SetMarkerStyle(kPlus)##kDot)
             points.Draw("PLsame")
             first = False
         legend = canvas.BuildLegend().SetFillColor(kWhite)
         legend.SetFillStyle(0)
         canvas.SaveAs("plot_"+str(xaxis)+"_"+str(yaxis)+".png")
         canvas.SaveAs("plot_"+str(xaxis)+"_"+str(yaxis)+".eps")
         canvas.SaveAs("plot_"+str(xaxis)+"_"+str(yaxis)+".pdf")
         canvas.SaveAs("plot_"+str(xaxis)+"_"+str(yaxis)+".root")
         canvas.SaveAs("plot_"+str(xaxis)+"_"+str(yaxis)+".C")
         del h
if __name__ == "__main__":
    stylefile = "/home/pseyfert/coding/root/official.C"
    if os.path.isfile(stylefile):
       from ROOT import gROOT
       gROOT.ProcessLine(".x "+stylefile)
    runstats()
