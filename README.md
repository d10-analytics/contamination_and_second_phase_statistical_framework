# A Statistical Framework for Contamination and Second Phase Quantification in Metal and Ceramic Materials

Metal and ceramic powders and solid materials are susceptible to contamination from multiple sources — cross-contamination between alloy systems, oxide and ceramic inclusions introduced during processing, and foreign particles picked up during handling, sieving, and recycling. This applies across industries including additive manufacturing, powder metallurgy, coatings, and any application where compositional purity matters. Despite this, no established methodology exists within the US and international standards framework (ASTM, ANSI, ISO) for quantifying contamination rates with statistical rigor — most published work stops at detection and classification rather than estimation of rates with known uncertainty.

This repository contains the code and reproducible analysis for a paper that presents a complete statistical framework for post-classification data: given tabulated phase identification results (particle counts, phase labels, area measurements per field), the framework provides formal statistical sampling design with pilot data and sample size estimation, contamination and second phase rate estimates with confidence intervals and standard errors, and a hypothesis testing framework for regulatory acceptance criteria. The framework applies equally to powder samples and solid (metallographic) cross-sections.

While this paper uses large-area EDS spectral imaging as the analytical method for phase identification, the statistical framework itself is method-agnostic — any technique that produces a labeled phase map (e.g., EBSD, Raman, XRF) can feed the same sampling, estimation, and hypothesis testing pipeline.

The raw data is hosted externally due to size constraints; a link will be added here once available.

## Analytical Workflow

1. A representative powder sample is mounted for SEM/EDS analysis and a large-area spectral image is collected. 
2. Phase identification software classifies each pixel into a compositionally distinct phase, producing a labeled phase map identifying the matrix alloy, any contaminant or second phases, and their spatial distributions.
3. **Stage 1 (Pilot):** A small preliminary dataset (e.g., a 2x2 grid of fields) is collected and analyzed to obtain an initial contamination estimate and its variance. A sample size calculation determines how much additional area must be collected to achieve a target precision.
4. **Stage 2 (Full Analysis):** The full dataset is collected at the calculated sample size, and the complete statistical analysis is performed: distinct contaminant phase counts, individual and overall contamination rates with confidence intervals, and hypothesis tests against acceptance criteria.

## Statistical Framework

The framework supports two parallel contamination metrics:

- **Particle count rate:** the number of contaminant particles per unit area (or per N total particles). Relevant when individual foreign particles are the concern.
- **Area fraction:** the total area occupied by contaminant phase(s) as a fraction of the measured area. Relevant when the volume of contamination matters — for example, oxide inclusion content affecting mechanical properties.

The hypothesis testing framework is designed for regulatory and quality control applications with conservative framing: the null hypothesis is that contamination equals or exceeds a specified threshold, and rejection constitutes statistical evidence that contamination is below the acceptance criterion at the chosen significance level. This conservative framing places the burden of proof on demonstrating cleanliness — the appropriate direction for safety-critical applications.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Development

After cloning, enable the repository hooks with `git config core.hooksPath .githooks`.
