from astroquery.ipac.ned import Ned
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import pyvo
import requests

class TimeoutSession(requests.Session):
    """
    Requests session that applies a default timeout to all requests.

    Parameters
    ----------
    timeout : float
        Default request timeout in seconds.
    """
    def __init__(self, timeout):
        super().__init__()
        self.timeout = timeout

    def request(self, *args, **kwargs):
        kwargs.setdefault("timeout", self.timeout)
        return super().request(*args, **kwargs)


def _get_ned_sources(RA_range, DEC_range, only_spec_z = True, timeout = 60, max_sources = 30):
    """
    Query NED for objects within a sky region and return matching sources as a DataFrame.

    Returns None if no sources are found in the requested region.
    """
    RA_center = (RA_range[0] + RA_range[1]) / 2
    DEC_center = (DEC_range[0] + DEC_range[1]) / 2
    RA_size = np.abs(np.diff(RA_range))[0]
    DEC_size = np.abs(np.diff(DEC_range))[0]

    session = TimeoutSession(timeout) # only to apply timeout
    service = pyvo.dal.TAPService("https://ned.ipac.caltech.edu/tap", session=session)
    query_cols = ['prefname', 'prefphytype', 'ra', 'dec', 'zflag', 'z']

    query = f"""
            SELECT TOP {max_sources}
                {", ".join(query_cols)}
            FROM objdir
            WHERE CONTAINS(
                POINT('J2000', ra, dec),
                BOX('J2000',{RA_center}, {DEC_center}, {RA_size},{DEC_size})
            ) = 1
            """
    try:
        result = service.search(query, maxrec=max_sources + 1)
    except Exception as e:
        print(f"❌ NED query failed: {e}")
        raise e

    if result is None:
        return None
    else:
        df = result.to_table().to_pandas()
        if "z" in df.columns:
            df = df[df["z"].notna()]
        if "prefphytype" in df.columns:
            df = df[~df["prefphytype"].isin(["GPair", "GTrpl", "GGroup", "GClstr", "QGroup"])]
        if only_spec_z:
            df = df[df["zflag"].str.startswith("S")]
            df = df[~df["zflag"].str.contains("?", regex=False)]
        df.rename(columns={"prefname": "object_id"}, inplace=True)
        df.rename(str.lower, axis='columns', inplace=True)
        df["object_id"] = df["object_id"].astype(str)

    return df

def _ned_cone_search(ra_center, dec_center, radius_arcmin=5, only_spec_z = False, timeout = 120):
    """
    Alternative method to search for nearby objects in NED. Currently not in use.
    """
    Ned.TIMEOUT = timeout
    # print("NED.TIMEOUT:",Ned.TIMEOUT)
    radius_deg = radius_arcmin/(60*60)
    pos = SkyCoord(ra=ra_center, dec=dec_center, unit=(u.degree, u.degree))
    # Query NED around that position

    try:
        result = Ned.query_region(pos, radius=radius_deg*u.degree)
    except Exception as e:
        print(f"❌ NED query failed: {e}")
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
    df_ned = _ned_cone_search(34.507, -0.4156, 0.1)
    # df_ned = get_ned_sources(151.6263, 2.12, 0.5)
    print(df_ned[["object_id", "ra", "dec", "z"]])
    quit()
    df_ned = get_ned_sources([151.6126, 151.6423], [2.1385,2.1487])
    print(df_ned[["object_id", "ra", "dec", "z"]])
    quit()

