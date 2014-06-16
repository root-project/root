
.PHONY: filter doxygen images clean

docdir="$(shell ./makedocdir.sh)"
imgdir="$(shell ./makeimgdir.sh)"
macdir="$(shell ./makemacrodir.sh)"

all: filter images doxygen

filter:
	`root-config --cxx` -o filter filter.cxx `root-config --libs --glibs --cflags`

images:
	./makeallimages.sh $(imgdir) $(macdir)

doxygen:
	doxygen

clean:
	rm -r $(macdir)
	rm -r $(imgdir)
	rm -r $(docdir)
	rm filter
