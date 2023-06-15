from typing import get_type_hints


def drjitstruct(cls):
    drjit_struct = {}

    type_hints = get_type_hints(cls)

    for name, ty in type_hints.items():
        drjit_struct[name] = ty
    cls.DRJIT_STRUCT = drjit_struct
    return cls
