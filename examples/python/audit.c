#define _GNU_SOURCE
#include <stdio.h>
#include <link.h>

unsigned int la_version(unsigned int version) {
    return version;
}

uintptr_t la_symbind64(Elf64_Sym *sym, unsigned int ndx,
                       uintptr_t *refcook, uintptr_t *defcook,
                       unsigned int *flags, const char *symname) {
    printf("[AUDIT] Binding symbol: %s\n", symname);
    return sym->st_value;  // Correct: return as uintptr_t
}

unsigned int la_objopen(struct link_map *map, Lmid_t lmid, uintptr_t *cookie) {
    printf("[AUDIT] Loaded object: %s\n", map->l_name);
    return LA_FLG_BINDTO | LA_FLG_BINDFROM;
}
