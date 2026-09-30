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

Early stage: archival photometry is being collected and the core binary-lens solver is not yet written. See the repo's issues/project board for current progress.

## Output

Results (grid search χ² maps per event) will be published openly on Zenodo, with anything scientifically notable submitted to [Research Notes of the AAS](https://journals.aas.org/research-notes/).

## Acknowledgments

This research made use of publicly available survey data. See each survey's own citation requirements (linked above) before publishing results derived from their data.
