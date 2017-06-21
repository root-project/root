_numpyinterface.so: _numpyinterface.cxx
	g++ -shared -fPIC _numpyinterface.cxx -o _numpyinterface.so `root-config --cflags --libs` -L `root-config --libdir` -I /usr/include/python2.7 -I /usr/lib64/python2.7/site-packages/numpy/core/include

all: _numpyinterface.cxx

clean:
	rm -f _numpyinterface.so
