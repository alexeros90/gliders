from astroquery.gama import GAMA
from ..utils import _check_ra_range_leap

def _get_gama_sources(ra_range, dec_range, timeout = 60, max_sources = 30):
    """
    Query the GAMA catalogue for objects within a sky region and return matching sources as a DataFrame.

    Returns None if no sources are found in the requested region.
    """

    if _check_ra_range_leap(ra_range):
        ra_line = f"AND (sa.RA BETWEEN {ra_range[0]} AND 360 OR sa.RA BETWEEN 0 AND {ra_range[1]}) "
    else:
        ra_line = f"AND sa.RA BETWEEN {ra_range[0]} AND {ra_range[1]} "

    sql = f"""
            SELECT SPECID, SURVEY, RA, DEC, Z, GAMA_NAME, DIST, IS_SBEST, IS_BEST
            FROM SpecAll AS sa
            WHERE sa.SURVEY = 'GAMA' 
            {ra_line}
            AND sa.DEC BETWEEN {dec_range[0]} AND {dec_range[1]}
            AND IS_SBEST = True
            AND IS_BEST = True
            LIMIT {max_sources}
            """
    GAMA.clear_cache()
    GAMA.timeout = timeout
    # print("GAMA.TIMEOUT:", GAMA.timeout)
    try:
        result = GAMA.query_sql(sql)

    except (ValueError, Exception) as e:
        if isinstance(e, ValueError) and "Results did not contain a result url" in str(e):
            # usually triggered when there are no sources in the region. Hence, for now we do not raise the error.
            return None
        print(f"❌ GAMA query failed: {e}")
        raise e

    if result is None:
        return None
    else:
        df = result.to_pandas()
        df.rename(columns={"SPECID": "object_id", "RA": "ra", "DEC": "dec", "Z": "z"}, inplace=True)
        df["object_id"] = df["object_id"].astype(str)

    return df

if __name__ == "__main__":
    df_gama = _get_gama_sources([030.56160,030.564], [-07.49153, -07.48])
    print(df_gama[["object_id", "ra", "dec", "z"]])
    quit()


