## \file
## \ingroup tutorial_dataframe
## \notebook
## Get information about the dataframe with the convenience method Describe.
##
## \macro_code
## \macro_output
##
## \date March 2021
## \author Stefan Wunsch (KIT, CERN)

import ROOT

# Create a dataframe
path = 'root://eospublic.cern.ch//eos/opendata/cms/derived-data/AOD2NanoAODOutreachTool/Run2012BC_DoubleMuParked_Muons.root'
df = ROOT.RDataFrame('Events', path)

# Describe the state of the dataframe.
# Note that this operation is not running the event loop.
# Describe returns a DFDescription object, which has e.g. a Print method. See its docs for more information.
df.Describe().Print()

# Build a small analysis studying the invariant mass of dimuon systems.
# See tutorial df102_NanoAODDimuonAnalysis for more information.
df = df.Filter('nMuon == 2')\
       .Filter('Muon_charge[0] != Muon_charge[1]')\
       .Define('Dimuon_mass', 'InvariantMass(Muon_pt, Muon_eta, Muon_phi, Muon_mass)')\
       .Filter('Dimuon_mass > 70')\
       .Range(1000)

# Trigger the event loop by asking for the mean of the dimuon mass.
print('\nApproximate mass of the Z boson: {:.2f} GeV\n'.format(
    df.Mean('Dimuon_mass').GetValue()))

# This time we ask for the `shortFormat`, which only prints a brief description of the dataset:
df.Describe().Print(shortFormat=True)
