ROOT_CONFIG=../../root-install/bin/root-config
# ROOT_CONFIG=root-config

PYTHON_INCLUDE_DIR=/cvmfs/cms.cern.ch/slc6_amd64_gcc530/external/python/2.7.11-oenich2/include/python2.7
# PYTHON_INCLUDE_DIR=/usr/include/python2.7

# PYTHON_LIBRARY=/cvmfs/cms.cern.ch/slc6_amd64_gcc530/cms/cmssw-patch/CMSSW_9_0_2_patch1/external/slc6_amd64_gcc530/lib/libpython2.7.so

NUMPY_INCLUDE_DIR=/cvmfs/cms.cern.ch/slc6_amd64_gcc530/external/py2-numpy/1.11.1-oenich2/lib/python2.7/site-packages/numpy-1.11.1-py2.7-linux-x86_64.egg/numpy/core/include
# NUMPY_INCLUDE_DIR=/usr/lib64/python2.7/site-packages/numpy/core/include

_numpyinterface.so: _numpyinterface.cxx
	g++ -shared -fPIC _numpyinterface.cxx -o _numpyinterface.so `$(ROOT_CONFIG) --cflags --libs` -L `root-config --libdir` -I $(PYTHON_INCLUDE_DIR) -I $(NUMPY_INCLUDE_DIR)

all: _numpyinterface.cxx

clean:
	rm -f _numpyinterface.so
