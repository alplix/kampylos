#pragma once
// A pause point for helper threads (2026-11-05; KAM-027).
//
// When the BOINC client suspends a task, the BOINC runtime stops only the thread that receives its timer signal
// (the main thread): threads the app created keep computing until their share of the work is done, using CPU
// that the client has meanwhile given to another task (KAM-027: the GPU apps' completion pool; and the seed pool
// of a multi-threaded CPU app). Helper threads therefore call kampylos_worker_pause() between two pieces of work
// (candidates, objective evaluations). Outside BOINC, and in the single-threaded apps, it does nothing; the BOINC
// wrappers point it at a wait that lasts while the client has the task suspended
// (kampylos_boinc_install_worker_pause() in boinc/kampylos_boinc_common.h). It only delays work, so it cannot
// change a result.
inline void (*&kampylos_worker_pause_fn())() { static void (*f)() = nullptr; return f; }
inline void kampylos_worker_pause() {
    if (void (*f)() = kampylos_worker_pause_fn()) f();
}
