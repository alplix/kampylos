#pragma once
// Stand-in for BOINC's lib/filesys.h (see boinc_api.h in this directory).
#include <cstdio>

FILE* boinc_fopen(const char* path, const char* mode);
int boinc_rename(const char* old_name, const char* new_name);
int boinc_delete_file(const char* path);
