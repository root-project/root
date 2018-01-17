#####################################
#
# 'DATA AND CATEGORIES' ROOT.RooFit tutorial macro #404
#
# Working with ROOT.RooCategory objects to describe discrete variables
#
#
#
# 07/2008 - Wouter Verkerke
#
# /

import ROOT


def rf404_categories():

    # C o n s t r u c t    a   c a t e g o r y   w i t h   l a b e l s
    # ----------------------------------------------------------------

    # Define a category with labels only
    tagCat = ROOT.RooCategory("tagCat", "Tagging category")
    tagCat.defineType("Lepton")
    tagCat.defineType("Kaon")
    tagCat.defineType("NetTagger-1")
    tagCat.defineType("NetTagger-2")
    tagCat.Print()

    # C o n s t r u c t    a   c a t e g o r y   w i t h   l a b e l s   a n d   i n d e c e s
    # ----------------------------------------------------------------------------------------

    # Define a category with explicitly numbered states
    b0flav = ROOT.RooCategory("b0flav", "B0 flavour eigenstate")
    b0flav.defineType("B0", -1)
    b0flav.defineType("B0bar", 1)
    b0flav.Print()

    # G e n e r a t e   d u m m y   d a t a  f o r   t a b u l a t i o n   d e m o
    # ----------------------------------------------------------------------------

    # Generate a dummy dataset
    x = ROOT.RooRealVar("x", "x", 0, 10)
    data = ROOT.RooPolynomial("p", "p", x).generate(
        ROOT.RooArgSet(x, b0flav, tagCat), 10000)

    # P r i n t   t a b l e s   o f   c a t e g o r y   c o n t e n t s   o f   d a t a s e t s
    # ------------------------------------------------------------------------------------------

    # ROOT.Tables are equivalent of plots for categories
    btable = data.table(b0flav)
    btable.Print()
    btable.Print("v")

    # Create table for subset of events matching cut expression
    ttable = data.table(tagCat, "x>8.23")
    ttable.Print()
    ttable.Print("v")

    # Create table for all (tagCat x b0flav) state combinations
    bttable = data.table(ROOT.RooArgSet(tagCat, b0flav))
    bttable.Print("v")

    # Retrieve number of events from table
    # Number can be non-integer if source dataset has weighed events
    nb0 = btable.get("B0")
    print "Number of events with B0 flavor is ", nb0

    # Retrieve fraction of events with "Lepton" tag
    fracLep = ttable.getFrac("Lepton")
    print "Fraction of events tagged with Lepton tag is ", fracLep

    # D e f i n i n g   r a n g e s   f o r   p l o t t i n g , i t t i n g   o n   c a t e g o r i e s
    # ------------------------------------------------------------------------------------------------------

    # Define named range as comma separated list of labels
    tagCat.setRange("good", "Lepton,Kaon")

    # Or add state names one by one
    tagCat.addToRange("soso", "NetTagger-1")
    tagCat.addToRange("soso", "NetTagger-2")

    # Use category range in dataset reduction specification
    goodData = data.reduce(ROOT.RooFit.CutRange("good"))
    goodData.table(tagCat).Print("v")


if __name__ == "__main__":
    rf404_categories()
