# ATLAS manuscript (Statistics in Medicine submission)

Compile with pdfLaTeX (run three times; no BibTeX step, references are in references.tex).

- main.tex         main paper with Appendices A to D (journal "Main Document")
- supplement.tex   standalone Supplementary Material, Sections S1 to S10
- atlas_arxiv.tex  main paper, appendices and supplement in one PDF (arXiv)

The data files, archived results and reference code cited in the text
(supplementary/, results_20260919/, resource_results/, reference_code/) are
packaged separately for the code and data repository.

Current main comparisons use Dynamic-DeepHit (optimized). In the separate data/code
package, see README.txt and supplementary/README.txt for the current integrated
results in supplementary/optimized_DDH_integrated_2026-11-02/. Original DDH results
are retained in Supplementary Section S10 and supplementary/original_DDH_results_2026-11-02/.

DRSA and Dynamic-DeepHit use their validation-selected optimized configurations.
The current integrated comparison data are in the data/code package under
supplementary/optimized_DRSA_integrated_2026-11-16/. Historical DRSA results
are excluded from the manuscript; prior research records remain archived.

## Approved submission-review update: 18 November 2026

Set the Overleaf main document to main.tex for the journal main paper,
supplement.tex for the standalone supplement, or atlas_arxiv.tex for the combined version.
This release incorporates approved review items 1-4 and 6-8: supplementary
numbering/pagination, propagation and estimation costs, hardware provenance,
author details and declarations, AI-use disclosure, figure legibility, and
the Delphi reference and MMLR Discussion refinement.

The compiled/ directory contains the corresponding verified review PDFs
(32-page main paper, 49-page supplement and 78-page combined paper).
These proof PDFs were built with Tectonic; pdfLaTeX remains the recommended
Overleaf compiler. The source_assets/ directory retains the editable S9 source.

Item 5 (the updated data/code release and repository deposition) remains
postponed. This ZIP is the manuscript/figure package only. The repository DOI
placeholder remains in the manuscript and must be completed before submission.
Coauthors should confirm the contribution, funding and competing-interest statements.
