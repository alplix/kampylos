// Fake BOINC runtime for testing the Kampylos wrappers as plain processes (see boinc_api.h).
// Files are plain files in the working directory. boinc_time_to_checkpoint() follows the real
// API's contract (true at most once per period, then re-armed by boinc_checkpoint_completed());
// the period comes from KAMPYLOS_STUB_CHECKPOINT_SECONDS (default 60, 0 = every call), so a
// resume test can make the app checkpoint after every seed.
#include "boinc_api.h"
#include "filesys.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

static double stub_period = 60.0;
static std::chrono::steady_clock::time_point stub_last;
static int stub_critical = 0;

int boinc_init_options(BOINC_OPTIONS*) {
    if (const char* p = getenv("KAMPYLOS_STUB_CHECKPOINT_SECONDS")) stub_period = atof(p);
    stub_last = std::chrono::steady_clock::now();
    return 0;
}

int boinc_finish(int status) {
    fprintf(stderr, "boinc_stub: boinc_finish(%d)\n", status);
    fflush(stderr);
    exit(status);
}

int boinc_resolve_filename(const char* virtual_name, char* physical_name, int len) {
    snprintf(physical_name, len, "%s", virtual_name);
    return 0;
}

int boinc_get_init_data(APP_INIT_DATA& d) { d = APP_INIT_DATA(); return 0; }

int boinc_fraction_done(double f) {
    FILE* fp = fopen("stub_fraction_done", "w");
    if (fp) { fprintf(fp, "%.6f\n", f); fclose(fp); }
    return 0;
}

int boinc_time_to_checkpoint() {
    double since = std::chrono::duration<double>(std::chrono::steady_clock::now() - stub_last).count();
    if (since < stub_period) return 0;
    boinc_begin_critical_section();
    return 1;
}

int boinc_checkpoint_completed() {
    stub_last = std::chrono::steady_clock::now();
    FILE* fp = fopen("stub_checkpoints", "a");
    if (fp) { fprintf(fp, "checkpoint\n"); fclose(fp); }
    boinc_end_critical_section();
    return 0;
}

void boinc_begin_critical_section() { stub_critical++; }
void boinc_end_critical_section() { if (stub_critical > 0) stub_critical--; }

FILE* boinc_fopen(const char* path, const char* mode) { return fopen(path, mode); }
int boinc_rename(const char* old_name, const char* new_name) { return rename(old_name, new_name); }
int boinc_delete_file(const char* path) { return remove(path); }
