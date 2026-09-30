// CLI driver for the binary-lens grid search over one archival event, one sub-range of the
// (log s, log q) grid at a time -- this sub-range restriction is exactly what will become one
// BOINC work unit: the caller passes which slice of the grid this run covers, so many
// volunteers can cover one event's full grid between them.
//
// Usage:
//   fit_event <photometry_file> <mag|flux> <log_s_min> <log_s_max> <n_s> \
//             <log_q_min> <log_q_max> <n_q> <out_file>
//
// Grid is n_s x n_q points, linearly spaced in log(s) and log(q) over the given ranges
// (inclusive). Writes one line per grid cell to out_file: log_s log_q chi2 t0 u0 tE alpha rho fs fb
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "lightcurve.h"
#include "pspl_prefit.h"
#include "binary_fit.h"

int main(int argc, char** argv) {
    if (argc != 10) {
        fprintf(stderr, "usage: %s <photometry_file> <mag|flux> <log_s_min> <log_s_max> <n_s> "
                         "<log_q_min> <log_q_max> <n_q> <out_file>\n", argv[0]);
        return 1;
    }
    std::string photfile = argv[1];
    bool input_is_mag = (strcmp(argv[2], "mag") == 0);
    double log_s_min = atof(argv[3]), log_s_max = atof(argv[4]);
    int n_s = atoi(argv[5]);
    double log_q_min = atof(argv[6]), log_q_max = atof(argv[7]);
    int n_q = atoi(argv[8]);
    std::string outfile = argv[9];

    std::vector<DataPoint> data;
    try {
        data = load_photometry(photfile, input_is_mag);
    } catch (const std::exception& e) {
        fprintf(stderr, "error loading %s: %s\n", photfile.c_str(), e.what());
        return 1;
    }
    if (data.size() < 20) {
        fprintf(stderr, "error: only %zu usable data points (need at least 20)\n", data.size());
        return 1;
    }
    fprintf(stderr, "Loaded %zu data points from %s\n", data.size(), photfile.c_str());

    VBMicrolensing vbm;

    PsplFit anchor = pspl_prefit(vbm, data);
    fprintf(stderr, "PSPL anchor: t0=%.4f u0=%.5f tE=%.4f chi2=%.3f\n",
            anchor.t0, anchor.u0, anchor.tE, anchor.chi2);

    FILE* out = fopen(outfile.c_str(), "w");
    if (!out) { fprintf(stderr, "cannot open %s for writing\n", outfile.c_str()); return 1; }
    fprintf(out, "# log_s log_q chi2 t0 u0 tE alpha rho fs fb\n");

    int total = n_s * n_q;
    int done = 0;
    for (int is = 0; is < n_s; is++) {
        double log_s = (n_s == 1) ? log_s_min : log_s_min + (log_s_max - log_s_min) * is / (n_s - 1);
        for (int iq = 0; iq < n_q; iq++) {
            double log_q = (n_q == 1) ? log_q_min : log_q_min + (log_q_max - log_q_min) * iq / (n_q - 1);

            BinaryFitResult r = fit_binary_multistart(
                vbm, data, log_s, log_q, anchor.t0, anchor.u0, anchor.tE
            );

            fprintf(out, "%.6f %.6f %.4f %.6f %.6f %.6f %.6f %.6e %.4f %.4f\n",
                    r.log_s, r.log_q, r.chi2, r.t0, r.u0, r.tE, r.alpha, r.rho, r.fs, r.fb);
            fflush(out);

            done++;
            fprintf(stderr, "[%d/%d] log_s=%.3f log_q=%.3f chi2=%.3f\n", done, total, log_s, log_q, r.chi2);
        }
    }
    fclose(out);
    fprintf(stderr, "Done: %d grid cells written to %s\n", total, outfile.c_str());
    return 0;
}
