from ROOT import TChain
import os

original_filename = "org.root"
def generate_testfilename(alg,level):
    return "size."+str(alg)+"."+str(level)+".root"

def check(org,alg,level):
    '''
    check if the file for an algorithm and level setting is a replicate of org

    Keyword arguments:
    org    -- the reference file
    alg    -- compression algorithm
    level  -- compression level
    '''
    verify = TChain("B02DD")
    # check if adding succeeds
    verify.Add(generate_testfilename(alg,level))
    allpassed = True
    for e in range(org.GetEntries()):
        org.GetEntry(e)
        verify.GetEntry(e)
        for b in org.GetListOfBranches():
            if getattr(org,b.GetName()) != getattr(verify,b.GetName()):
                allpassed = False
            if not allpassed:
                break
        if not allpassed:
            break
    return allpassed

def get_orgfile():
  if not os.path.isfile(original_filename):
    raise IOError("reference file not found")
  org = TChain("B02DD")
  org.Add(original_filename)
  return org


if __name__=="__main__":
  org = get_orgfile()
  allpassed = True
  for alg in [1,2,4,5,6,7]:
      for level in range(1,10):
          try:
              allpassed = allpassed and check(org,alg,level)
          except:
              raise

  if allpassed:
      print "test passed"
  else:
      print "test failed"

