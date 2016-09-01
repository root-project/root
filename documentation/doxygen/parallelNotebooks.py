try:
    import dagger
    from joblib import Parallel, delayed
    import subprocess
    import multiprocessing
    import os

    outDir = os.environ["DOXYGEN_NOTEBOOK_PATH_PARALLEL"]

    inputs = subprocess.check_output(["grep",  "-r",  "-l", "/// \\\\notebook\|## \\\\notebook",  
             os.path.expandvars("$DOXYGEN_SOURCE_DIRECTORY/tutorials")]).split()

    dependenciesgraph = dagger.dagger()

    for element in inputs:
        dependenciesgraph.add(element)

    def root(tutName):
        return os.path.expandvars("$ROOTSYS/tutorials/")+tutName

    dependenciesgraph.add(root("math/testUnfold5d.C"),[root("math/testUnfold5c.C")])
    dependenciesgraph.add(root("math/testUnfold5c.C"),[root("math/testUnfold5b.C")])
    dependenciesgraph.add(root("math/testUnfold5b.C"),[root("math/testUnfold5a.C")])
    dependenciesgraph.add(root("xml/xmlreadfile.C"),[root("xml/xmlnewfile.C")])
    dependenciesgraph.add(root("roofit/rf503_wspaceread.C"),[root("roofit/rf502_wspacewrite.C")])
    dependenciesgraph.add(root("io/readCode.C"),[root("io/importCode.C")])
    dependenciesgraph.add(root("fit/fit1.C"),[root("hist/fillrandom.C")])
    dependenciesgraph.add(root("fit/myfit.C"),[root("fit/fitslicesy.C")])
    dependenciesgraph.add(root("foam/foam_demopers.C"),[root("foam/foam_demo.C")])
    dependenciesgraph.add(root("tree/staff.C"),[root("tree/cernbuild.C")])
    dependenciesgraph.add(root("tree/cernstaff.C"),[root("tree/cernbuild.C")])
    dependenciesgraph.add(root("hist/hbars.C"),[root("tree/cernbuild.C")])
    dependenciesgraph.add(root("pyroot/ntuple1.py"),[root("pyroot/hsimple.py")])
    dependenciesgraph.add(root("pyroot/h1draw.py"),[root("pyroot/hsimple.py")])
    dependenciesgraph.add(root("pyroot/fit1.py"),[root("pyroot/fillrandom.py")])
    dependenciesgraph.add(root("tmva/TMVAClassificationApplication.C"),[root("tmva/TMVAClassification.C")])
    dependenciesgraph.add(root("tmva/TMVAClassificationCategory.C"),[root("tmva/TMVAClassification.C")])
    dependenciesgraph.add(root("tmva/TMVAClassificationCategoryApplication.C"),[root("tmva/TMVAClassificationCategor")])
    dependenciesgraph.add(root("tmva/TMVAMulticlass.C"),[root("tmva/TMVAMultipleBackgroundExample.C")])
    dependenciesgraph.add(root("tmva/TMVAMulticlassApplication.C"),[root("tmva/TMVAMulticlass.C")])
    dependenciesgraph.add(root("tmva/TMVARegressionApplication.C"),[root("tmva/TMVARegression.C")])

    for node in dependenciesgraph.nodes:
        dependenciesgraph.stale(node)

    dependenciesgraph.run()

    iterator = dependenciesgraph.iter()

    newinputs = []
    while len(iterator)>0:
        todo = iterator.next(10000)
        newinputs.append(todo)
        # print todo
        for element in todo:
            iterator.remove(element)

    for i in newinputs:
        print i

    def processInput(inputFile):
        subprocess.call(['python', './converttonotebook.py', inputFile, outDir])
    num_cores = multiprocessing.cpu_count()

    def parallel(input):
        Parallel(n_jobs=num_cores,verbose=100)(delayed(processInput)(i) for i in input)

    for input in newinputs:
        parallel(input)

except:
    pass