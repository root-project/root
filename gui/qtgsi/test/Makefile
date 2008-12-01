SUBDIRS := example1 example2

.PHONY: $(SUBDIRS) all lib clean

lib: $(SUBDIRS)

all: lib

clean:
	@for dir in $(SUBDIRS); do cd $$dir ; $(MAKE) clean ROOTCONFIG=../../../../bin/root-config; cd .. ; done

$(SUBDIRS):
	@cd $@; $(MAKE) all
