# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

################################################################
# CMakeLists.txt file for building ROOT pythonizations libraries
################################################################

if(dataframe)
    list(APPEND PYROOT_EXTRA_PYTHON_SOURCES
        ROOT/_pythonization/_rdf_utils.py
        ROOT/_pythonization/_rdataframe.py
        ROOT/_pythonization/_rdfdescription.py
        ROOT/_pythonization/_rdf_conversion_maps.py
        ROOT/_pythonization/_rdf_pyz.py
        ROOT/_pythonization/_rdisplay.py
        ROOT/_pythonization/_rdf_namespace.py
        ROOT/_pythonization/_ml_dataloader.py)
endif()

if(roofit)
    list(APPEND PYROOT_EXTRA_PYTHON_SOURCES
        ROOT/_pythonization/_roofit/__init__.py
        ROOT/_pythonization/_roofit/_rooabscollection.py
        ROOT/_pythonization/_roofit/_rooabsdata.py
        ROOT/_pythonization/_roofit/_rooabspdf.py
        ROOT/_pythonization/_roofit/_rooabsreallvalue.py
        ROOT/_pythonization/_roofit/_rooabsreal.py
        ROOT/_pythonization/_roofit/_rooarglist.py
        ROOT/_pythonization/_roofit/_rooargset.py
        ROOT/_pythonization/_roofit/_roocategory.py
        ROOT/_pythonization/_roofit/_roodatahist.py
        ROOT/_pythonization/_roofit/_roodataset.py
        ROOT/_pythonization/_roofit/_roodecays.py
        ROOT/_pythonization/_roofit/_roogenfitstudy.py
        ROOT/_pythonization/_roofit/_rooglobalfunc.py
        ROOT/_pythonization/_roofit/_roojsonfactorywstool.py
        ROOT/_pythonization/_roofit/_roomcstudy.py
        ROOT/_pythonization/_roofit/_roomsgservice.py
        ROOT/_pythonization/_roofit/_rooprodpdf.py
        ROOT/_pythonization/_roofit/_roorealvar.py
        ROOT/_pythonization/_roofit/_roosimultaneous.py
        ROOT/_pythonization/_roofit/_roosimwstool.py
        ROOT/_pythonization/_roofit/_roostats.py
        ROOT/_pythonization/_roofit/_roovectordatastore.py
        ROOT/_pythonization/_roofit/_rooworkspace.py
        ROOT/_pythonization/_roofit/_utils.py)
endif()

if(tmva)
    list(APPEND PYROOT_EXTRA_PYTHON_SOURCES
        ROOT/_pythonization/_tmva/_crossvalidation.py
        ROOT/_pythonization/_tmva/_dataloader.py
        ROOT/_pythonization/_tmva/_factory.py
        ROOT/_pythonization/_tmva/__init__.py
        ROOT/_pythonization/_tmva/_rbdt.py
        ROOT/_pythonization/_tmva/_rtensor.py
        ROOT/_pythonization/_tmva/_tree_inference.py
        ROOT/_pythonization/_tmva/_utils.py
        ROOT/_pythonization/_tmva/_gnn.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/__init__.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/parser.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/__init__.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/batchnorm.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/binary.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/concat.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/conv.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/dense.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/elu.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/flatten.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/identity.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/layernorm.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/leaky_relu.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/permute.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/pooling.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/reshape.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/relu.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/rnn.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/selu.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/sigmoid.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/softmax.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/swish.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/tanh.py)
endif()

set(py_sources
  #ROOT/JsMVA/DataLoader.py
  #ROOT/JsMVA/Factory.py
  #ROOT/JsMVA/JPyInterface.py
  #ROOT/JsMVA/JsMVAMagic.py
  #ROOT/JsMVA/OutputTransformer.py
  #ROOT/JsMVA/__init__.py
  ROOT/_application.py
  ROOT/_asan.py
  ROOT/_facade.py
  ROOT/__init__.py
  ROOT/_jupyroot/__init__.py
  ROOT/_jupyroot/helpers/__init__.py
  ROOT/_jupyroot/helpers/cppcompleter.py
  ROOT/_jupyroot/helpers/handlers.py
  ROOT/_jupyroot/helpers/utils.py
  ROOT/_jupyroot/html/__init__.py
  ROOT/_jupyroot/html/cpphighlighter.py
  ROOT/_jupyroot/kernel/__init__.py
  ROOT/_jupyroot/kernel/rootkernel.py
  ROOT/_jupyroot/kernel/utils.py
  ROOT/_jupyroot/kernel/magics/__init__.py
  ROOT/_jupyroot/kernel/magics/cppmagic.py
  ROOT/_jupyroot/kernel/magics/jsrootmagic.py
  ROOT/_jupyroot/magics/__init__.py
  ROOT/_jupyroot/magics/cppmagic.py
  ROOT/_jupyroot/magics/jsrootmagic.py
  ROOT/_numbadeclare.py
  ROOT/_pythonization/__init__.py
  ROOT/_pythonization/_cppinstance.py
  ROOT/_pythonization/_drawables.py
  ROOT/_pythonization/_generic.py
  ROOT/_pythonization/_memory_utils.py
  ROOT/_pythonization/_pyz_utils.py
  ROOT/_pythonization/_rfile.py
  ROOT/_pythonization/_rntuple.py
  ROOT/_pythonization/_runtime_error.py
  ROOT/_pythonization/_rvec.py
  ROOT/_pythonization/_stl_vector.py
  ROOT/_pythonization/_tarray.py
  ROOT/_pythonization/_tbrowser.py
  ROOT/_pythonization/_tclass.py
  ROOT/_pythonization/_tclonesarray.py
  ROOT/_pythonization/_tcollection.py
  ROOT/_pythonization/_tcolor.py
  ROOT/_pythonization/_tcomplex.py
  ROOT/_pythonization/_tcontext.py
  ROOT/_pythonization/_tdirectory.py
  ROOT/_pythonization/_tdirectoryfile.py
  ROOT/_pythonization/_tefficiency.py
  ROOT/_pythonization/_tentrylist.py
  ROOT/_pythonization/_teventlist.py
  ROOT/_pythonization/_tf1.py
  ROOT/_pythonization/_tf2.py
  ROOT/_pythonization/_tf3.py
  ROOT/_pythonization/_tfile.py
  ROOT/_pythonization/_tfilemerger.py
  ROOT/_pythonization/_tformula.py
  ROOT/_pythonization/_tcanvas.py
  ROOT/_pythonization/_tgraph.py
  ROOT/_pythonization/_tgraph2d.py
  ROOT/_pythonization/_th1.py
  ROOT/_pythonization/_th2.py
  ROOT/_pythonization/_th3.py
  ROOT/_pythonization/_thistpainter.py
  ROOT/_pythonization/_titer.py
  ROOT/_pythonization/_tobject.py
  ROOT/_pythonization/_tobjstring.py
  ROOT/_pythonization/_tscatter.py
  ROOT/_pythonization/_tseqcollection.py
  ROOT/_pythonization/_tstring.py
  ROOT/_pythonization/_tstyle.py
  ROOT/_pythonization/_ttree.py
  ROOT/_pythonization/_tvector3.py
  ROOT/_pythonization/_tvectort.py
  ROOT/_pythonization/_uhi/__init__.py
  ROOT/_pythonization/_uhi/tags.py
  ROOT/_pythonization/_uhi/indexing.py
  ROOT/_pythonization/_uhi/plotting.py
  ROOT/_pythonization/_uhi/serialization.py
  ${PYROOT_EXTRA_PYTHON_SOURCES}
)

ROOT_PYTHON_PACKAGE(ROOT SOURCES ${py_sources})

# The Python version at build time should be easy to figure out from Python, so
# we can raise an exception if the used Python version is not compatible.
set(python_version_file "${localruntimedir}/ROOT/_python_version.py")
file(WRITE "${python_version_file}" "_root_python_version = \"${Python3_VERSION}\"\n")
install(FILES "${python_version_file}" DESTINATION "${CMAKE_INSTALL_PYTHONDIR}")
