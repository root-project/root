
SRC= \
    TMemStat_Technical_Overview.xml

DOCBOOK=TMemStat_Technical_Overview.xml

HTML_OUT_DIR=./

#IMGS= \
	important.png\
	tip.png\
	warning.png\
	note.png
#IMG_DIR = img

HTML_OUT= \
          TMemStat_Technical_Overview.html

CSS_FILES=docbook.css

XMLTO=xmlto

#XMLTO_PARAM=-vv -o $(HTML_OUT_DIR) xhtml -m config.xsl
#XMLTO_PARAM_NOCHUNKS=-o $(HTML_OUT_DIR) xhtml-nochunks -m config.xsl
XMLTO_PARAM=-o $(HTML_OUT_DIR) xhtml-nochunks -m config.xsl

all: gen_html

$(HTML_OUT_DIR):
	@echo "Creating output directory for html doc [$(HTML_OUT_DIR)]..."
	@mkdir $(HTML_OUT_DIR)

$(HTML_OUT_DIR)/%.css:
	@echo "Using custom CSS..."
	@cp $(CSS_FILES) $(HTML_OUT_DIR)/

$(HTML_OUT_DIR)/%.html: $(SRC)
	@echo "Generating html doc..."
	$(XMLTO) $(XMLTO_PARAM) $(DOCBOOK)

#$(HTML_OUT_DIR)/%.png:
#	@echo "Copying images..."
#	@cp ${addprefix $(IMG_DIR)/, $(IMGS)} $(HTML_OUT_DIR)/

html_files: ${addprefix $(HTML_OUT_DIR)/, $(HTML_OUT)}
css_files: ${addprefix $(HTML_OUT_DIR)/, $(CSS_FILES)}
#img_files: ${addprefix $(HTML_OUT_DIR)/, $(IMGS)}

#gen_html: $(HTML_OUT_DIR) css_files img_files html_files
gen_html: $(HTML_OUT_DIR) css_files html_files

clean:
	@-rm -f index.html
