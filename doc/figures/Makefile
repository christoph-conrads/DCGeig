LATEX := pdflatex -file-line-error -halt-on-error -synctex=1

EXT      := tex
SOURCES  := $(wildcard *.$(EXT))
TARGETS  := $(patsubst %.$(EXT), %.pdf, $(SOURCES))


all: $(TARGETS)


%.pdf: %.tex
	$(LATEX) $<



.PHONY: clean all
clean:
	$(RM) *.aux *.bbl *.bcf *.blg *-blx.bib *.ilg *.log *.nav *.nlo *.nls *.out *.run.xml *.snm *.synctex.gz *.toc *.xml
	$(RM) *.glg *.glo *.gls *.ind *.ist *.sbl *.sym
	$(RM) *.fdb_latexmk *.fls
	$(RM) *-eps-converted-to.pdf
