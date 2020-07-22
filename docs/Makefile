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
	sphinx-apidoc -f --templatedir=$(TEMPLATEDIR) -o $(SOURCEDIR) ../datasetinsights/ ../datasetinsights/data/datasets/coco.py ../datasetinsights/data/datasets/cityscapes.py ../datasetinsights/data/datasets/kitti.py ../datasetinsights/data/datasets/nuscenes.py ../datasetinsights/data/datasets/__pycache__ ../datasetinsights/data/datasets/nyudepth.py ../datasetinsights/data/datasets/groceries_real.py ../datasetinsights/cli.py ../datasetinsights/torch_distributed.py ../datasetinsights/visual.py ../datasetinsights/configs/ ../datasetinsights/estimators/ ../datasetinsights/evaluation_metrics/ ../datasetinsights/font/

html: apidoc
	sphinx-build -b html $(SOURCEDIR) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML doumentation pages are in $(BUILDDIR)/html."
