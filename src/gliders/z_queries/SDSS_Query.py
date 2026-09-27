from astroquery.sdss import SDSS
from astropy import coordinates as coords
from astropy.coordinates import SkyCoord
import astropy.units as u
from ..utils import square_region_from_cone_search

def _get_sdss_sources(RA_range, DEC_range, data_release = 19, timeout = 60):
    """Query SDSS sources within a sky region.
    The search is performed using an SDSS cone search centered on the
    requested region. Because SDSS limits cone-search radii to 3 arcmin,
    larger regions are truncated to this maximum radius. Results are
    subsequently filtered to match the requested rectangular bounds."""
    point_1 = SkyCoord(ra=RA_range[0]*u.degree, dec=DEC_range[0]*u.degree, frame='icrs')
    point_2 = SkyCoord(ra=RA_range[1]*u.degree, dec=DEC_range[1]*u.degree, frame='icrs')
    radius = point_1.separation(point_2).arcmin/2
    if radius > 3:
        print("Maximum allowed search radius is 3 arcmin. Truncating to 3 arcmin.")
        radius = 3
        # SDSS does not allow searches greater than 3 arcminutes
    RA_center = (RA_range[0] + RA_range[1]) / 2
    DEC_center = (DEC_range[0] + DEC_range[1]) / 2
    sources = _sdss_cone_search(RA_center, DEC_center, radius_arcmin=radius, data_release = data_release, timeout = timeout)

    sources = square_region_from_cone_search(sources, RA_range, DEC_range)

    return sources


def _sdss_cone_search(RA_center, DEC_center, radius_arcmin, data_release = 19, photo=False, timeout = 60):
    """
    Query SDSS for all sources within an RA/Dec box (approx via cone).

    Parameters
    ----------
    ra_min, ra_max, dec_min, dec_max : float
        Bounds in degrees.
    photo : bool
        If True, return photometric sources. If False, return only spectroscopic.

    Returns
    -------
    pandas.DataFrame
        RA, DEC, z, specClass, etc.
    """
    pos = coords.SkyCoord(RA_center, DEC_center, unit="deg", frame='icrs')

    try:
        result = SDSS.query_region(pos, radius=f"{radius_arcmin} arcmin", spectro=(not photo),
                                   data_release=data_release, timeout = timeout, cache = False)
    except Exception as e:
        print(f"❌ SDSS query failed: {e}")
        raise e
        # return None

    if result is None:
        return None
    else:
        df = result.to_pandas()
        if "z" in df.columns:
            df = df[df["z"].notna()]
        df.rename(columns={"objid": "object_id"}, inplace=True)
        df["object_id"] = df["object_id"].astype(str)

    return df


if __name__ == "__main__":
    df_sdss = _sdss_cone_search(229.526, 42.745, 0.5, data_release = 19)
    print(df_sdss[["object_id", "ra", "dec", "z"]])
    df_sdss = _get_sdss_sources([229.50, 229.54], [42.72, 42.76], data_release = 19)
    print(df_sdss[["object_id", "ra", "dec", "z"]])
    quit()

