from pipeline.run_pipeline import meters_per_pixel

def test_area_conversion_positive():
    lat = 12.9716
    zoom = 19

    mpp = meters_per_pixel(lat, zoom)
    area_px = 1000  # arbitrary pixel area
    area_m2 = area_px * (mpp ** 2)

    assert area_m2 > 0, "Converted area must be positive"
