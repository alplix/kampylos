#pragma once
// Minimal stand-in for BOINC's api/boinc_api.h, enough to build the Kampylos BOINC wrappers
// without a BOINC checkout and run them as plain processes (tests/test_boinc_resume.sh).
// Only the functions the wrappers call are provided; see boinc_stub.cpp for their behaviour.
#include <string>

struct BOINC_OPTIONS {
    int main_program = 0;
    int check_heartbeat = 0;
    int handle_process_control = 0;
    int send_status_msgs = 0;
    int direct_process_action = 0;
    int multi_thread = 0;
    int multi_process = 0;
    int normal_thread_priority = 0;
};

struct APP_INIT_DATA {
    int gpu_device_num = -1;
    int gpu_opencl_dev_index = -1;
    double checkpoint_period = 60;
};

inline void boinc_options_defaults(BOINC_OPTIONS& o) { o = BOINC_OPTIONS(); }
int boinc_init_options(BOINC_OPTIONS*);
int boinc_finish(int status);
int boinc_temporary_exit(int delay, const char* reason = 0, bool is_notice = false);
int boinc_resolve_filename(const char* virtual_name, char* physical_name, int len);
int boinc_get_init_data(APP_INIT_DATA&);
int boinc_fraction_done(double);
int boinc_time_to_checkpoint();
int boinc_checkpoint_completed();
void boinc_begin_critical_section();
void boinc_end_critical_section();
