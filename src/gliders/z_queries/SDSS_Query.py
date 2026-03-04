from astroquery.sdss import SDSS
from astropy import coordinates as coords
from astropy.coordinates import SkyCoord
import astropy.units as u

def get_sdss_square_region(RA_range, DEC_range, data_release = 19, timeout = 60):
    point_1 = SkyCoord(ra=RA_range[0]*u.degree, dec=DEC_range[0]*u.degree, frame='icrs')
    point_2 = SkyCoord(ra=RA_range[1]*u.degree, dec=DEC_range[1]*u.degree, frame='icrs')
    radius = point_1.separation(point_2).arcmin/2
    if radius > 3:
        radius = 3
        # SDSS does not allow searches greater than 3 arcminutes
    RA_center = (RA_range[0] + RA_range[1]) / 2
    DEC_center = (DEC_range[0] + DEC_range[1]) / 2
    sources = get_sdss_sources(RA_center, DEC_center, radius_arcmin=radius, data_release = data_release, timeout = timeout)

    if sources is not None:
        sources = sources[(sources['ra'] >= RA_range[0]) & (sources['ra'] <= RA_range[1]) &
                          (sources['dec'] >= DEC_range[0]) & (sources['dec'] <= DEC_range[1])]
    return sources


def get_sdss_sources(RA_center, DEC_center, radius_arcmin, data_release = 19, photo=False, timeout = 60):
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
    SDSS.TIMEOUT = timeout
    # print("SDSS.TIMEOUT:",SDSS.TIMEOUT)

    try:
        result = SDSS.query_region(pos, radius=f"{radius_arcmin} arcmin",
                                   spectro=(not photo), data_release=data_release)
    except Exception as e:
        print(f"❌ SDSS Query failed: {e}")
        return None

    if isinstance(result, type(None)):
        return None
    else:
        df = result.to_pandas()
        df.rename(columns={"objid": "object_id"}, inplace=True)
        df["object_id"] = df["object_id"].astype(str)

    # Keep redshifts if available
    if "z" in df.columns:
        df = df[df["z"].notna()]
    return df


if __name__ == "__main__":
    df_sdss = get_sdss_sources(229.526, 42.745, 0.5, data_release = 19)
    print(df_sdss[["object_id", "ra", "dec", "z"]])
    df_sdss = get_sdss_square_region([229.50, 229.54], [42.72, 42.76], data_release = 19)
    print(df_sdss[["object_id", "ra", "dec", "z"]])
    quit()

