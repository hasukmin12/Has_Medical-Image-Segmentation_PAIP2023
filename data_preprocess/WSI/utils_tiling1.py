import os
import numpy as np
from pathlib import Path
# from numpy.lib.shape_base import tile
from openslide import ImageSlide, open_slide
from openslide.deepzoom import DeepZoomGenerator
from multiprocessing import Process, JoinableQueue
from utils_filter import noise_patch, good_tile


class TileWorker(Process):
    def __init__(self, queue, slidepath: str, tile_size: int, overlap: int, limit_bounds: bool, use_filter: bool,
                 quality=None):
        Process.__init__(self, name='TileWorker')
        self.daemon = True
        self._queue = queue
        self._slidepath = slidepath
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._use_filter = use_filter
        self._quality = quality
        self._slide = None

    def run(self):
        self._slide = open_slide(self._slidepath)
        last_associated = None
        dz = self._get_dz()
        while True:
            data = self._queue.get()
            if data is None:
                self._queue.task_done()
                break
            associated, level, address, outfile = data
            if last_associated != associated:
                dz = self._get_dz(associated)
                last_associated = associated
            tile = dz.get_tile(level, address)

            if not self._use_filter:
                tile.save(outfile)
            elif not noise_patch(np.array(tile)):
                tile.save(outfile)

            self._queue.task_done()

    def _get_dz(self, associated=None):
        if associated is not None:
            image = ImageSlide(self._slide.associated_images[associated])
        else:
            image = self._slide
        return DeepZoomGenerator(
            image,
            tile_size=self._tile_size,
            overlap=self._overlap,
            limit_bounds=self._limit_bounds)


class DeepZoomImageTiler(object):
    def __init__(self, dz, basename, tiledir, _format, associated, queue, level):
        self._dz = dz
        self._basename = basename
        self._tiledir = tiledir
        self._format = _format
        self._associated = associated
        self._queue = queue
        self._level = level
        self._processed = 0

    def run(self):
        self._write_tiles()
        # self._write_dzi()

    def _write_tiles(self):
        cols, rows = self._dz.level_tiles[self._level]
        for row in range(1, rows - 1):
            for col in range(1, cols - 1):
                tilename = os.path.join(self._tiledir, f"{self._basename}-{row}_{col}.{self._format}")
                if not os.path.exists(tilename):
                    self._queue.put((self._associated, self._level, (col, row), tilename))
                # self._tile_done()
                self._processed += 1

    # def _tile_done(self):
    #     self._processed += 1
    #     count, total = self._processed, self._dz.tile_count

    # def _write_dzi(self):
    #     with open(f"{self._basename}.dzi", "w") as f:
    #         f.write(self.get_dzi())

    def get_dzi(self):
        return self._dz.get_dzi(self._format)


def delete_light_patches(_dir: str, ext: str = 'jpg') -> None:
    patches = list(Path(_dir).glob(f"**/*.{ext}"))
    for p in patches:
        if os.path.getsize(p) < 1024*10:
            os.remove(p)

def delete_light_patches_seg(_dir: str, ext: str = 'jpg') -> None:
    patches = list(Path(_dir).glob(f"**/*.{ext}"))
    for p in patches:
        if os.path.getsize(p) < 1024*10:
            os.remove(p)


def delete_light_patches_new(_dir, ext, threshold) :
    patches = list(Path(_dir).glob(f"**/*.{ext}"))
    for p in patches:
        if os.path.getsize(p) < threshold:
            os.remove(p)


def count_saved_patches(_dir: str,ext: str) -> int:
    return len(list(Path(_dir).glob(f"**/*.{ext}")))


def saved_patches_ratio(
        slide_file: str,
        tile_size: int,
        overlap: int,
        tile_dir: str,
        resolution_factor: int,
        limit_bounds: bool = False,
        ext: str = 'jpg') -> float:
    slide = open_slide(slide_file)
    tiles = DeepZoomGenerator(
        slide,
        tile_size=tile_size,
        overlap=overlap,
        limit_bounds=limit_bounds
    )
    max_level = tiles.level_count - 1
    level = max_level - int(np.log(resolution_factor) / np.log(2))
    cols, rows = tiles.level_tiles[level]
    total_patches = tiles.tile_count

    saved_patches = list(Path(tile_dir).glob(f"**/*.{ext}"))

    return len(saved_patches)/total_patches


def save_patches(
        slide_file: str,
        output_path: str,
        resolution_factor: float,
        tile_size: int,
        overlap: int,
        ext: str = 'jpg',
        use_filter: bool = True,
        limit_bounds: bool = False,
        workers: int = 20) -> None:
        
    tile_size = tile_size - (2*overlap)
    
    try:
        queue = JoinableQueue(2 * workers)
        slide = open_slide(slide_file)
    except Exception as e:
        print(f"Could not open {slide_file}\n")
        print(e)
        return

    W, H = slide.dimensions
    if (W == 0) or (H == 0):
        print(f"Could not open {slide_file}\n")
        return

    slide_name = Path(slide_file).stem
    for _ in range(workers):
        TileWorker(queue, slide_file, tile_size=tile_size, overlap=overlap, limit_bounds=limit_bounds,
                   use_filter=use_filter).start()

    tiles = DeepZoomGenerator(
        slide,
        tile_size=tile_size,
        overlap=overlap,
        limit_bounds=limit_bounds
    )

    max_level = tiles.level_count - 1
    level = max_level
    cols, rows = tiles.level_tiles[level]

    tiler = DeepZoomImageTiler(
        dz=tiles,
        basename=slide_name,
        tiledir=Path(output_path),
        _format=ext,
        associated=None,
        queue=queue,
        level=level
    )

    tiler.run()

    for _ in range(workers):
        queue.put(None)
    queue.close()
    queue.join()

    return level, cols, rows


def save_patches_no_multiprocessor(
        slide_file: str,
        output_path: str,
        resolution_factor: int,
        tile_size: int,
        overlap: int,
        ext: str,
        use_filter: bool = True,
        limit_bounds: bool = False) -> None:

    tile_size = tile_size - (2*overlap)
    # Opening slide and getting assure it exists
    try:
        slide = open_slide(slide_file)
    except TypeError:
        print(f"Could not open {slide_file}")
        return
    W, H = slide.dimensions

    if (W == 0) or (H == 0):
        print(f"Could not open {slide_file}")
        return

    slide_name = Path(slide_file).stem
    tiles = DeepZoomGenerator(
        slide, 
        tile_size=tile_size,
        overlap=overlap, 
        limit_bounds=limit_bounds)

    # Translating resolution factor to zoom level
    max_level = tiles.level_count - 1
    level = max_level - int(np.log(resolution_factor) / np.log(2))

    # Saving tiles
    cols, rows = tiles.level_tiles[level]
    for row in range(rows):
        for col in range(cols):
            tile = tiles.get_tile(level, (col, row))
            if not use_filter:
                tile_name = f"{slide_name}-{row}_{col}.{ext}"
                tile_path = Path(output_path).joinpath(tile_name)
                if not tile_path.exists():
                    tile.save(str(tile_path))
            elif not noise_patch(np.array(tile)):
                tile_name = f"{slide_name}-{row}_{col}.{ext}"
                tile_path = Path(output_path).joinpath(tile_name)
                if not tile_path.exists():
                    tile.save(str(tile_path))
    slide.close()
    return
