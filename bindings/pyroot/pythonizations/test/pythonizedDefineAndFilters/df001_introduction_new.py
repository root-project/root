import ROOT
import numpy as np


def fill_tree(treeName, fileName):
    # TODO: How to do this directly from python
    df = ROOT.RDataFrame(10)
    df.PyDefine("b1", "(double) rdfentry_")\
        .PyDefine("b2", "(int) rdfentry_*rdfentry_").Snapshot(treeName, fileName)

fileName = "df001_introduction_py.root"
treeName = "myTree"
fill_tree(treeName, fileName)

d = ROOT.RDataFrame(treeName, fileName)

def cutb1(b1):
    return b1<5

def cutb1b2(b1, b2):
    return bool(b2%2) and b1<4

entries1 = d.Filter(cutb1)\
            .Filter(cutb1b2)\
            .Count()

print('{} entries passed all filters'.format(entries1.GetValue()))

entries2 = d.Filter(lambda b1: bool(b1 < 5)).Count();
print('{} entries passed all filters'.format(entries2.GetValue()))

b1b2_cut = d.Filter(cutb1b2)
minVal = b1b2_cut.Min('b1')
maxVal = b1b2_cut.Max('b1')
meanVal = b1b2_cut.Mean('b1')
nonDefmeanVal = b1b2_cut.Mean("b2")
print('The mean is always included between the min and the max: {0} <= {1} <= {2}'.format(minVal.GetValue(), meanVal.GetValue(), maxVal.GetValue()))
 
hist = d.Filter(cutb1).Histo1D('b1')
print('Filled h {0} times, mean: {1}'.format(hist.GetEntries(), hist.GetMean()))
 
cutb1_result = d.Filter(cutb1);
cutb1b2_result = d.Filter(cutb1b2);
cutb1_cutb1b2_result = cutb1_result.Filter(cutb1b2)
 
# Now we want to count:
evts_cutb1_result = cutb1_result.Count()
evts_cutb1b2_result = cutb1b2_result.Count()
evts_cutb1_cutb1b2_result = cutb1_cutb1b2_result.Count()
 
print('Events passing cutb1: {}'.format(evts_cutb1_result.GetValue()))
print('Events passing cutb1b2: {}'.format(evts_cutb1b2_result.GetValue()))
print('Events passing both: {}'.format(evts_cutb1_cutb1b2_result.GetValue()))
 
entries_sum = d.PyDefine('sum', 'b2 + b1') \
               .Filter('sum > 4.2') \
               .Count()
print(entries_sum.GetValue())