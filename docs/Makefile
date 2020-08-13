# Makefile for Sphinx documentation

BUILDDIR      = build
SOURCEDIR     = source
TEMPLATEDIR   = $(SOURCEDIR)/_templates

.PHONY: help clean html

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"

clean:
	rm -rf $(BUILDDIR)
	find $(SOURCEDIR) -name "*.rst" | grep -v index | xargs rm -f

apidoc:
	sphinx-apidoc -f --templatedir=$(TEMPLATEDIR) -o $(SOURCEDIR) ../datasetinsights/ ../datasetinsights/ststs/visualization ../datasetinsights/dashboard.py ../datasetinsights/datasets/protos
html: apidoc
	sphinx-build -b html $(SOURCEDIR) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML doumentation pages are in $(BUILDDIR)/html."
