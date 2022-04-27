## \file
## \ingroup tutorial_roofit
## \notebook
##
## 'ORGANIZATION AND SIMULTANEOUS FITS' RooFit tutorial macro #508
##
## RooArgSet and RooArgList tools and tricks
##
## \macro_code
##
## \date February 2018
## \authors Clemens Lange, Wouter Verkerke (C version)

from __future__ import print_function
import ROOT


# Create dummy objects
# ---------------------------------------

# Create some variables
a = ROOT.RooRealVar("a", "a", 1, -10, 10)
b = ROOT.RooRealVar("b", "b", 2, -10, 10)
c = ROOT.RooRealVar("c", "c", 3, -10, 10)
d = ROOT.RooRealVar("d", "d", 4, -10, 10)
x = ROOT.RooRealVar("x", "x", 0, -10, 10)
c.setError(0.5)
a.setConstant()
b.setConstant()

# Create a category
e = ROOT.RooCategory("e", "e")
e.defineType("sig")
e.defineType("bkg")

# Create a pdf
g = ROOT.RooGaussian("g", "g", x, a, b)

# Creating, killing RooArgSets
# -------------------------------------------------------

# A ROOT.RooArgSet is a set of RooAbsArg objects. Each object in the set must have
# a unique name

# Set constructors exists with up to 9 initial arguments
s = ROOT.RooArgSet(a, b)

# At any time objects can be added with add()
s.add(e)

# Add up to 9 additional arguments in one call
# s.add(ROOT.RooArgSet(c, d))
s.add(c)
s.add(d)

# Sets can contain any type of RooAbsArg, pdf and functions
s.add(g)

# Remove element d
s.remove(d)

# Accessing RooArgSet contents
# -------------------------------------------------------

# You can look up objects by name
aptr = s.find("a")

# Construct a subset by name
subset1 = s.selectByName("a,b,c")

# Construct asubset by attribute
subset2 = s.selectByAttrib("Constant", ROOT.kTRUE)

# Construct the subset of overlapping contents with another set
s1 = ROOT.RooArgSet(a, b, c)
s2 = ROOT.RooArgSet(c, d, e)
subset3 = s1.selectCommon(s2)

# Owning RooArgSets
# ---------------------------------

# Create a RooArgSet that owns its components
# A set either owns all of its components or none,
# so once addOwned() is used, add() can no longer be
# used and will result in an error message

ac = a.clone("a")
bc = b.clone("b")
cc = c.clone("c")

s3 = ROOT.RooArgSet()
# s3.addOwned(ROOT.RooArgSet(ac, bc, cc))
s3.addOwned(ac)
s3.addOwned(bc)
s3.addOwned(cc)

# Another possibility is to add an owned clone
# of an object instead of the original
# s3.addClone(ROOT.RooArgSet(d, e, g))
s3.addClone(d)
s3.addClone(e)
s3.addClone(g)

# A clone of a owning set is non-owning and its
# contents is owned by the originating owning set
sclone = s3.Clone("sclone")

# To make a clone of a set and its contents use
# the snapshot method
sclone2 = s3.snapshot()

# If a set contains function objects, the head node
# is cloned in a snapshot. To make a snapshot of all
# servers of a function object do as follows. The result
# of a RooArgSet snapshot with deepCloning option is a set
# of cloned objects, all their clone (recursive) server
# dependencies, together form a self-consistent
# set that is free of external dependencies

sclone3 = s3.snapshot(ROOT.kTRUE)

# Set printing
# ------------------------

# Inline printing only show list of names of contained objects
print("sclone = ", sclone)

# Plain print shows the same, by name of the set
sclone.Print()

# Standard printing shows one line for each item with the items name, name
# and value
sclone.Print("s")

# Verbose printing adds each items arguments, and 'extras' as defined by
# the object
sclone.Print("v")

# Using RooArgLists
# ---------------------------------

# List constructors exists with up to 9 initial arguments
l = ROOT.RooArgList(a, b, c, d)

# Lists have an explicit order and allow multiple arguments with the same
# name
l.add(ROOT.RooArgList(a, b, c, d))

# Access by index is provided
arg4 = l.at(4)
