from astroquery.ipac.ned import Ned
import astropy.units as u
from astropy.coordinates import SkyCoord
# from ..utils import check_ra_range_leap
from gliders.utils import check_ra_range_leap

def get_ned_square_region(RA_range, DEC_range, only_spec_z = True, timeout = 120):
    point_1 = SkyCoord(ra=RA_range[0]*u.degree, dec=DEC_range[0]*u.degree, frame='icrs')
    point_2 = SkyCoord(ra=RA_range[1]*u.degree, dec=DEC_range[1]*u.degree, frame='icrs')
    radius = point_1.separation(point_2).arcmin/2
    RA_center = (RA_range[0] + RA_range[1]) / 2
    DEC_center = (DEC_range[0] + DEC_range[1]) / 2
    sources = get_ned_sources(RA_center, DEC_center, radius_arcmin=radius, only_spec_z=only_spec_z, timeout = timeout)
    leap_ra = check_ra_range_leap(RA_range)

    if sources is not None:
        if not leap_ra:
            sources = sources[(sources['ra'] >= RA_range[0]) & (sources['ra'] <= RA_range[1]) &
                              (sources['dec'] >= DEC_range[0]) & (sources['dec'] <= DEC_range[1])]
        else:
            sources = sources[(((sources['ra'] >= RA_range[0]) & (sources['ra'] <= 360)) |
                               ((sources['ra'] <= RA_range[1]) & (sources['ra'] >= 0))) &
                              (sources['dec'] >= DEC_range[0]) & (sources['dec'] <= DEC_range[1])]

    return sources

def get_ned_sources(ra_center, dec_center, radius_arcmin=5, only_spec_z = False, timeout = 120):
    """
    Query NED for all sources within a rectangular region or approximate circle.

    NED only supports circular (cone) searches, so for a box we sample the center.

    Parameters
    ----------
    ra_min, ra_max, dec_min, dec_max : float
        Region bounds in degrees.
    radius_arcmin : float
        Search radius around region center (in arcminutes).

    Returns
    -------
    pandas.DataFrame
        NED sources with position and redshift (if available).
    """
    Ned.TIMEOUT = timeout
    # print("NED.TIMEOUT:",Ned.TIMEOUT)
    radius_deg = radius_arcmin/60
    pos = SkyCoord(ra=ra_center, dec=dec_center, unit=(u.deg, u.deg))
    # Query NED around that position

    try:
        result = Ned.query_region(pos, radius=radius_deg*u.deg)
    except Exception as e:
        print(f"❌ NED Query failed: {e}")
        raise e

    # Only keep sources with redshift info
    df = result.to_pandas()
    if "Redshift" in df.columns:
        df = df[df["Redshift"].notna()]
        if "pretype" in df.columns:
            df = df[~df["pretype"].isin(["GPair","GClstr"])]
        if "Type" in df.columns:
            df = df[~df["Type"].isin(["GPair","GClstr"])]
        if only_spec_z:
            df = df[df["Redshift Flag"].str.startswith("S")]
            df = df[~df["Redshift Flag"].str.contains("?", regex=False)]
        df.rename(columns={"Redshift": "z", "Object Name": "object_id"}, inplace=True)
        df.rename(str.lower, axis='columns', inplace = True)
        df["object_id"] = df["object_id"].astype(str)
    return df

if __name__ == "__main__":
    # Example region near 150° < RA < 151°, 2° < Dec < 3°
    df_ned = get_ned_sources(34.507, -0.4156, 0.1)
    # df_ned = get_ned_sources(151.6263, 2.12, 0.5)
    print(df_ned[["object_id", "ra", "dec", "z"]])
    quit()
    df_ned = get_ned_square_region([151.6126, 151.6423], [2.1385,2.1487])
    print(df_ned[["object_id", "ra", "dec", "z"]])
    quit()

