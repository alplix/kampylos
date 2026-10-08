# Kampylos

*Kampylos* (Greek: καμπύλος, "bent" / "curved") — a distributed-computing project for exhaustive binary-lens gravitational microlensing modeling.

## The problem

Binary-lens microlensing events (a background star's light bent by a foreground *pair* of masses — a star with a companion star, or a star with a planet) require an expensive grid search over the lens parameters (mass ratio, projected separation, source trajectory angle) to find every viable solution. The χ² surface is often multimodal — several distinct parameter combinations can fit the same light curve almost equally well — and the standard workflow (grid search → refine → posterior sampling) is computationally expensive enough that, in practice, it's usually done by hand for one event at a time, with real risk of missing a real degenerate solution nobody thought to check.

Three public ground-based surveys (OGLE, MOA, KMTNet) have cumulatively alerted on tens of thousands of microlensing events since the 1990s, and only a fraction of the ones showing binary-lens (anomalous) light curves have ever been given this kind of exhaustive treatment — most published analyses focus on whichever solution was easiest to find, not a full accounting of the parameter space. The Roman Space Telescope (launched August 2026, science operations starting ~2027) is expected to add thousands more binary-lens events per year from its own microlensing survey, at a rate current manual/semi-manual methods can't keep up with.

## The approach

Volunteer-donated compute (via a BOINC app, sharing infrastructure with the [bitboinc](https://bitboinc.athena.org.tr/) project) runs an exhaustive grid search over the binary-lens parameter space for each archived event, evaluating the forward-modeled magnification curve (binary-lens equation, solved via the standard degree-5 complex polynomial root-finding used throughout the microlensing literature) against the real public photometry at every grid point, and reports back a full χ² map — not just one "best" answer, but every viable local minimum.

## Data sources

Public archival photometry from:
- [OGLE Early Warning System](https://ogle.astrouw.edu.pl/ogle4/ews/ews.html) (OGLE-III/IV, 2002–present)
- [MOA Alert Archive](https://moaprime.massey.ac.nz/moaarchive) (MOA-I/II, 2000–present)
- [KMTNet Data Archive](https://kmtnet.kasi.re.kr/ulens/) (2015–present, public after a 1-year proprietary period)

## Status

Live and running. The binary-lens solver (grid search + local optimization, using [VBMicrolensing](https://github.com/valboz/VBMicrolensing) for the magnification calculation) is deployed as a real BOINC app across CPU (12 platforms: Linux/Windows/macOS x86_64 and ARM, Android, and several less common architectures) and GPU backends (CUDA, OpenCL, Apple Metal, and MUSA), sharing bitboinc's volunteer compute pool. Several hundred archived OGLE/MOA events have been run through the full grid search so far, with new events being added continuously from the public archives listed above.

Every completed grid is checked against a short automated screen (complete grid coverage, planetary-range mass ratio, a sane flux-fit solution, a genuine interior χ² minimum rather than an edge-of-grid artifact, and a cross-check against the NASA Exoplanet Archive's already-published microlensing planets) before a human looks at it — this catches obvious non-detections and already-known results automatically, so review time goes toward the handful of events that actually need it.

## Grid convention and result format

Each grid cell fixes the binary separation s (in Einstein radii) and mass ratio q. Work units created since 2026-10-08 carry an explicit `log10` marker on their input line (`<mag|flux> s_min s_max n_s q_min q_max n_q log10`) and use a log10 grid: log10 s in [-1, 1] (11 points, so s = 1 is on the grid) and log10 q in [-5, 0] (10 points). Earlier work units have no marker and used natural logarithms, so their grid "log_q in [-4, 0]" covered q from about 0.018 to 1 only. Older datasets keep that meaning.

A result file starts with two header lines:

```
# grid=log10 log10_s log10_q chi2 t0 u0 tE alpha rho fs fb dchi2_single max_pt_dchi2 top3_dchi2 n_pts50 n_nights50 n_nights5pct
# anchor format=2 backend=... npts=... pspl_chi2=... fspl_chi2=... fspl_ld_chi2=... best=... chi2=... chi2dof=... ...
```

The anchor line records the best single-lens fit: point source, uniform finite source, or linearly limb-darkened finite source. The `dchi2_*` / `n_*` columns describe how each cell's improvement over that single-lens fit is distributed over the data points and nights (see `src/kampylos_wu.h`). The fs/fb flux parameters are the exact linear least-squares solution. Before 2026-10-08 they were ridge-regularised, which inflated chi2.

Results from before 2026-10-08 have a single header line, `# log_s log_q chi2 t0 u0 tE alpha rho fs fb`: a natural-log grid, ridge-biased chi2, and no single-lens anchor.

## Output

Completed grid search results are published openly on [Zenodo](https://zenodo.org), one dataset per event, as they clear review — see the project's public pages at [bitboinc.athena.org.tr](https://bitboinc.athena.org.tr) for the full list and current candidates. Anything that survives review as a genuine, previously-unreported binary-lens signal gets submitted to [Research Notes of the AAS](https://journals.aas.org/research-notes/).

## Acknowledgments

This research made use of publicly available survey data. See each survey's own citation requirements (linked above) before publishing results derived from their data.
