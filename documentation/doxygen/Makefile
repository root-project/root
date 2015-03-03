
.PHONY: filter doxygen clean

docdir="$(shell ./makedocdir.sh)"
imgdir="$(shell ./makeimgdir.sh)"
macdir="$(shell ./makemacrodir.sh)"

all: filter doxygen

filter:
	`root-config --cxx` -o filter filter.cxx `root-config --libs --glibs --cflags`

doxygen:
	doxygen

clean:
	rm -r $(macdir)
	rm -r $(imgdir)
	rm -r $(docdir)
	rm filter
