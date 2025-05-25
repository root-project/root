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

    dependenciesgraph.add("analysis/unfold/testUnfold5d.C",["analysis/unfold/testUnfold5c.C"])
    dependenciesgraph.add("analysis/unfold/testUnfold5c.C",["analysis/unfold/testUnfold5b.C"])
    dependenciesgraph.add("analysis/unfold/testUnfold5b.C",["analysis/unfold/testUnfold5a.C"])
    dependenciesgraph.add("io/xml/xmlreadfile.C",["io/xml/xmlnewfile.C"])
    dependenciesgraph.add("roofit/roofit/rf503_wspaceread.C",["roofit/roofit/rf502_wspacewrite.C"])
    dependenciesgraph.add("io/readCode.C",["io/importCode.C"])
    dependenciesgraph.add("math/fit/fit1.C",["hist/hist001_TH1_fillrandom.C"])
    dependenciesgraph.add("math/fit/myfit.C",["math/fit/fitslicesy.C"])
    dependenciesgraph.add("math/foam/foam_demopers.C",["math/foam/foam_demo.C"])
    dependenciesgraph.add("io/tree/tree502_staff.C",["io/tree/tree500_cernbuild.C"])
    dependenciesgraph.add("io/tree/tree501_cernstaff.C",["io/tree/tree500_cernbuild.C"])
    dependenciesgraph.add("hist/hist006_TH1_bar_charts.C",["io/tree/tree500_cernbuild.C"])
    dependenciesgraph.add("io/tree/ntuple1.py",["hsimple.py"])
    dependenciesgraph.add("math/fit/fit1.py",["hist/hist001_TH1_fillrandom.py"])
    dependenciesgraph.add("machine_learning/TMVAClassificationApplication.C",["machine_learning/TMVAClassification.C"])
    dependenciesgraph.add("machine_learning/TMVAClassificationCategory.C",["machine_learning/TMVAClassification.C"])
    dependenciesgraph.add("machine_learning/TMVAClassificationCategoryApplication.C",["machine_learning/TMVAClassificationCategory.C"])
    dependenciesgraph.add("machine_learning/TMVAMulticlass.C",["machine_learning/TMVAMultipleBackgroundExample.C"])
    dependenciesgraph.add("machine_learning/TMVAMulticlassApplication.C",["machine_learning/TMVAMulticlass.C"])
    dependenciesgraph.add("machine_learning/TMVARegressionApplication.C",["machine_learning/TMVARegression.C"])

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
