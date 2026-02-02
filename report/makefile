# =========================
# LaTeX Makefile
# =========================

MAIN := main
OUTDIR := pdflatex

LATEX := pdflatex
LATEXFLAGS := -interaction=nonstopmode -halt-on-error -output-directory=$(OUTDIR)

BIBTEX := bibtex

PDF := $(OUTDIR)/$(MAIN).pdf
TEX := $(MAIN).tex

.PHONY: all run clean distclean FORCE

all: run

run: $(PDF)
	@echo "Built: $(PDF)"

FORCE:

# QUI: FORCE è prerequisito del PDF, quindi viene sempre ricompilato
$(PDF): FORCE $(TEX) | $(OUTDIR)
	@echo "Compiling $(TEX) -> $(PDF)"
	$(LATEX) $(LATEXFLAGS) $(TEX)

	@if grep -q "Citation" "$(OUTDIR)/$(MAIN).log" 2>/dev/null || ls *.bib >/dev/null 2>&1; then \
		echo "Running BibTeX..."; \
		(cd $(OUTDIR) && $(BIBTEX) $(MAIN) || true); \
	fi

	$(LATEX) $(LATEXFLAGS) $(TEX)

	@if grep -q "Rerun to get cross-references right" "$(OUTDIR)/$(MAIN).log" 2>/dev/null; then \
		echo "Rerun requested by LaTeX..."; \
		$(LATEX) $(LATEXFLAGS) $(TEX); \
	fi

$(OUTDIR):
	@mkdir -p $(OUTDIR)

clean:
	@echo "Cleaning auxiliary files in $(OUTDIR)/"
	@rm -f $(OUTDIR)/*.aux $(OUTDIR)/*.bbl $(OUTDIR)/*.blg $(OUTDIR)/*.log \
	       $(OUTDIR)/*.out $(OUTDIR)/*.toc $(OUTDIR)/*.lof $(OUTDIR)/*.lot \
	       $(OUTDIR)/*.fls $(OUTDIR)/*.fdb_latexmk $(OUTDIR)/*.synctex.gz \
	       $(OUTDIR)/*.nav $(OUTDIR)/*.snm $(OUTDIR)/*.vrb

distclean: clean
	@echo "Removing PDF and output dir"
	@rm -f $(PDF)
	@rmdir $(OUTDIR) 2>/dev/null || true
