from pathlib import Path
from typing import Mapping, Sequence, Any


from fits_io.image_reader import get_reader
from fits_io.metadata import build_imagej_metadata
from fits_io.writer import save_tiff



def main(img_path: Path, save_path: Path, channel_labels: str | Sequence[str] | None = None, custom_metadata: Mapping[str, Any] | None = None, *, compression: str | None = 'zlib') -> None:
    # check if exp was processed/registered

    # get reader
    reader = get_reader(img_path)
    # read image and get metadata
    meta = build_imagej_metadata(reader, channel_labels=channel_labels, custom_metadata=custom_metadata)
    # write tiff with metadata and reader
    save_tiff(reader.get_array(), save_path, meta, compression=compression)



if __name__ == '__main__':
    from time import time

    t1 = time()
    
    img_path = Path('/home/ben/Docker_mount/Test_images/tiff/Run2/c2z25t23v1_tif.tif')
    r = get_reader(img_path)
    print(r.custom_metadata)
    # print(r.resolution)
    # print(r.interval)
    # print(r.axes)
    save_path = Path('/home/ben/Docker_mount/Test_images/tiff/Run2/test.tiff')
    main(img_path, save_path, channel_labels=["GFP", "mCherry"], custom_metadata={'Some_other_metadata': {'key1':'value1','key2':'value2'}})
    print(f"Done in {time()-t1:.2f} seconds")
    
    reader = get_reader(save_path)
    print('____________________')
    print(reader.resolution)
    print(reader.interval)
    print(reader.axes)
    print(reader.channel_labels)
    print(reader.custom_metadata)
    print(f'Done in {time()-t1:.2f} seconds')
