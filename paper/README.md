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

## Pre-Submission Notes

- Submit the source package rather than a PDF-only upload so arXiv can compile and stamp the TeX source.
- Review the arXiv-compiled PDF before finalizing; arXiv submissions must compile to a usable PDF before announcement.
- Candidate categories to evaluate in the arXiv UI: `cs.PF` for performance, `cs.DC` for systems/parallel-computing characterization, and `cs.CL` as a possible cross-list because the evaluated system serves a 70B-class language model.
- Suggested comment field after final compile: `Technical report; 70B-class AWQ inference on one NVIDIA L20; includes 24h soak, repeated fixed-shape runs, energy, and quality characterization.`
- Keep the claim narrow: this is a single-L20 AWQ serving characterization, not a full BF16/FP16 quality-retention or runtime-ablation paper.
