import math
from pipeline.run_pipeline import buffer_radius_pixels

def test_buffer_radius_increases():
    lat = 12.9716
    zoom = 19

    r_1200 = buffer_radius_pixels(1200, lat, zoom)
    r_2400 = buffer_radius_pixels(2400, lat, zoom)

    assert r_2400 > r_1200, "2400 sq.ft buffer should be larger than 1200 sq.ft buffer"
