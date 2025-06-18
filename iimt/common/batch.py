from typing import Union
from tqdm import tqdm

def get_batch(eles, batch_size:int = 32):
    length = len(eles)
    for i in tqdm(range(0, length, batch_size), total=round(length/batch_size)):
        yield eles[i:min(i+batch_size, length)]

def flatten(nested_eles):
    assert(isinstance(nested_eles[0], Union[list, tuple])), nested_eles[0]
    if len(nested_eles[0]) > 0:
        assert(not isinstance(nested_eles[0][0], Union[list, tuple])), 'flatten() only support one depth flatten.'
    output = []
    for eles in nested_eles:
        output += eles
    return output

def unflatten(flat_eles, original_nested_eles):
    assert(sum([len(eles) for eles in original_nested_eles]) == len(flat_eles)), (sum([len(eles) for eles in original_nested_eles]), len(flat_eles))
    output, start = [], 0
    for subeles in original_nested_eles:
        output.append(flat_eles[start:start+len(subeles)])
        start += len(subeles)
    return output