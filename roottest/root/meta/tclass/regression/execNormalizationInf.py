import os

# Importing ROOT before cppyy ensures that the cppyy-backend library can be
# found automatically via mechanism implemented in the ROOT module.
import ROOT

import cppyy

cppyy.gbl.std.map('string','string')()
