import gc

def remove_key(dict_in: dict, key: str) -> dict:
    dict_out = dict(dict_in)
    del dict_out[key]
    return dict_out

def shift_integers_to_zero(sequence: tuple[int,...]) -> tuple[int,...]:
    # (a, a + k) -> (0, k)
    offset = sequence[0]
    return offset_integers(sequence, -offset)

def offset_integers(sequence: tuple[int,...],
                    offset: int) -> tuple[int,...]:
    if offset == 0:
        return sequence
    else:
        return tuple(number + offset for number in sequence)

def cleanup_attributes(obj,
                       attrs: list[str] | str,
                       placeholder: list=None) -> None:

    if not isinstance(attrs, (list, tuple, set)):
        attrs = [attrs]

    for i, kw in enumerate(attrs):
        if not isinstance(kw, str):
            raise TypeError(f"Expect *args to be a collection of strings. Got '{type(kw)}'")

    for _, kw in enumerate(attrs):
        if hasattr(obj, kw) and getattr(obj, kw) is not None:
            delattr(obj, kw)
            setattr(obj, kw, placeholder)
        gc.collect()

    # if hasattr(obj, "_alias") and obj._alias:
    #     template = f"'{obj._alias}'"
    # else:
    #     template = f"object of class '{type(obj).__name__}'"
    #
    # print(f"\t Cleaned up attributes {repr(list(attrs))} for {template}.")