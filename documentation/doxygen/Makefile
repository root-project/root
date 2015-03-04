
.PHONY: filter doxygen clean

docdir="$(shell . ./makedocdir.sh)"
export OUTPUT_DIRECTORY=$(docdir)

all: filter doxygen

filter:
	`root-config --cxx` -o filter filter.cxx `root-config --libs --glibs --cflags`

doxygen:
	doxygen

clean:
	rm -r $(docdir)
	rm filter
	rm c1*
	rm stdout.dat
