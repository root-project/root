
.PHONY: filter doxygen clean

docdir="$(shell ./makedocdir.sh)"

all: filter doxygen

filter:
	`root-config --cxx` -o filter filter.cxx `root-config --libs --glibs --cflags`

doxygen:
	doxygen

clean:
	rm -r $(docdir)
	rm filter
	rm c1*
	rm *.png
	rm stdout.dat
