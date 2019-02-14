try:
    import dagger
    from joblib import Parallel, delayed
    import subprocess
    import multiprocessing
    import os

    outDir = os.environ["DOXYGEN_NOTEBOOK_PATH_PARALLEL"]

    inputs = [input.replace("../../tutorials/", "") for input in subprocess.check_output(["grep",  "-r",  "-l", "/// \\\\notebook\|## \\\\notebook", os.path.expandvars("$DOXYGEN_SOURCE_DIRECTORY/tutorials")]).split()]

    dependenciesgraph = dagger.dagger()

    for element in inputs:
        dependenciesgraph.add(element)

    dependenciesgraph.add("math/testUnfold5d.C",["math/testUnfold5c.C"])
    dependenciesgraph.add("math/testUnfold5c.C",["math/testUnfold5b.C"])
    dependenciesgraph.add("math/testUnfold5b.C",["math/testUnfold5a.C"])
    dependenciesgraph.add("xml/xmlreadfile.C",["xml/xmlnewfile.C"])
    dependenciesgraph.add("roofit/rf503_wspaceread.C",["roofit/rf502_wspacewrite.C"])
    dependenciesgraph.add("io/readCode.C",["io/importCode.C"])
    dependenciesgraph.add("fit/fit1.C",["hist/fillrandom.C"])
    dependenciesgraph.add("fit/myfit.C",["fit/fitslicesy.C"])
    dependenciesgraph.add("foam/foam_demopers.C",["foam/foam_demo.C"])
    dependenciesgraph.add("tree/staff.C",["tree/cernbuild.C"])
    dependenciesgraph.add("tree/cernstaff.C",["tree/cernbuild.C"])
    dependenciesgraph.add("hist/hbars.C",["tree/cernbuild.C"])
    dependenciesgraph.add("pyroot/ntuple1.py",["pyroot/hsimple.py"])
    dependenciesgraph.add("pyroot/h1draw.py",["pyroot/hsimple.py"])
    dependenciesgraph.add("pyroot/fit1.py",["pyroot/fillrandom.py"])
    dependenciesgraph.add("tmva/TMVAClassificationApplication.C",["tmva/TMVAClassification.C"])
    dependenciesgraph.add("tmva/TMVAClassificationCategory.C",["tmva/TMVAClassification.C"])
    dependenciesgraph.add("tmva/TMVAClassificationCategoryApplication.C",["tmva/TMVAClassificationCategory.C"])
    dependenciesgraph.add("tmva/TMVAMulticlass.C",["tmva/TMVAMultipleBackgroundExample.C"])
    dependenciesgraph.add("tmva/TMVAMulticlassApplication.C",["tmva/TMVAMulticlass.C"])
    dependenciesgraph.add("tmva/TMVARegressionApplication.C",["tmva/TMVARegression.C"])
 
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
        print(i)

    def processInput(inputFile):
        subprocess.call([sys.executable,
                         './converttonotebook.py', 
                         os.path.join(os.environ['DOXYGEN_SOURCE_DIRECTORY'], 'tutorials', inputFile), 
                         outDir])

    num_cores = multiprocessing.cpu_count()

    def parallel(input):
        Parallel(n_jobs=num_cores,verbose=100)(delayed(processInput)(i) for i in input)

    for input in newinputs:
        parallel(input)

except:
    print('Parallel notebooks converter failed!!')
    pass
