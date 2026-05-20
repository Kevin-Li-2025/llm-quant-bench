# Paper Artifacts

This directory contains the arXiv-style technical report draft:

- `single_l20_70b_awq_serving.tex`: main LaTeX source.
- `single_l20_70b_awq_serving.bbl`: generated bibliography for arXiv upload.
- `references.bib`: editable BibTeX references.
- `single_l20_70b_awq_serving.pdf`: compiled preview PDF.
- `single_l20_70b_awq_serving_arxiv_source.tar.gz`: source package containing the main `.tex`, `.bbl`, and `.bib`.

Local build command:

```bash
cd paper
TECTONIC_CACHE_DIR=/private/tmp/tectonic-cache tectonic single_l20_70b_awq_serving.tex
```

For arXiv, upload the source package and review the arXiv-compiled PDF before final submission.
