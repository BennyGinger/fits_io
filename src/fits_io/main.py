from pathlib import Path
from typing import Mapping, Sequence, Any

import numpy as np

from fits_io.image_reader import ImageReader, get_reader
from fits_io.metadata import build_imagej_metadata
from fits_io.writer import save_tiff



def convert_to_fits_tif(img_path: Path, *, save_path: Path | None = None, channel_labels: str | Sequence[str] | None = None, custom_metadata: Mapping[str, Any] | None = None, compression: str | None = 'zlib') -> None:
    # check if exp was processed/registered

    # get reader
    reader = get_reader(img_path)
    
    # read image and get metadata
    meta = build_imagej_metadata(reader, channel_labels=channel_labels, custom_metadata=custom_metadata)
    
    # Generate save path(s)
    gsave_path = _generate_save_path(reader, save_path)
    
    # Get the image array(s)
    arrays = reader.get_array()
    if isinstance(arrays, np.ndarray):
        arrays = [arrays]
    
    # write tiff with metadata and reader
    if len(arrays) != len(gsave_path):
        raise ValueError(f"Got {len(arrays)} arrays but {len(gsave_path)} save paths")

    for array, path in zip(arrays, gsave_path):
        save_tiff(array, path, meta, compression=compression)

def _generate_save_path(img_reader: ImageReader, save_path: Path | None = None) -> list[Path]:
    nb_arrays = img_reader.series_number
    
    if save_path is not None and nb_arrays == 1:
        return [save_path]
    
    # If path were given but multiple arrays, just implement '..._sx.tif' depending on number of arrays
    if save_path is not None and nb_arrays > 1:
        return [save_path.with_stem(f"{save_path.stem}_s{idx+1}") for idx in range(nb_arrays)]
    
    # If no path were given, generate paths
    base_name = img_reader.img_path.stem
    parent_dir = img_reader.img_path.parent
    save_name = 'fits_array.tif'
    
    save_paths: list[Path] = []
    for idx in range(nb_arrays):
        save_dir = parent_dir / f"{base_name}_s{idx+1}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_paths.append(save_dir / save_name)
        
    return save_paths
        

if __name__ == '__main__':
    from time import time

    t1 = time()
    
    img_path = Path('/home/ben/Docker_mount/Test_images/nd2/Run2copy/c2z25t23v1_nd2.nd2')
    convert_to_fits_tif(img_path, channel_labels=["GFP", "mCherry"], custom_metadata={'Some_other_metadata': {'key1':'value1','key2':'value2'}})
    print(f"Done in {time()-t1:.2f} seconds")
    
    new_path = Path('/home/ben/Docker_mount/Test_images/nd2/Run2copy/c2z25t23v1_nd2_s1/fits_array.tif')
    reader = get_reader(new_path)
    print('____________________')
    print(reader.resolution)
    print(reader.interval)
    print(reader.axes)
    print(reader.channel_labels)
    print(reader.custom_metadata)
    print(type(reader.custom_metadata))
    print(f'Done in {time()-t1:.2f} seconds')
