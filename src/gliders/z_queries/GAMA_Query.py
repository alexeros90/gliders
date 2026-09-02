from astroquery.gama import GAMA
from ..utils import check_ra_range_leap

def get_gama_sources(ra_range, dec_range, timeout = 60, max_sources = 20):

    if check_ra_range_leap(ra_range):
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

    try:
        GAMA.timeout = timeout
        # print("GAMA.TIMEOUT:", GAMA.timeout)
        result = GAMA.query_sql(sql)
        df = result.to_pandas()
        df.rename(columns={"SPECID": "object_id", "RA": "ra", "DEC": "dec", "Z": "z"}, inplace=True)
        df["object_id"] = df["object_id"].astype(str)
        return df
    except:
        # print(f"❌ GAMA Query failed: {e}")
        return None


if __name__ == "__main__":
    df_gama = get_gama_sources([030.56160,030.564], [-07.49153, -07.48])
    print(df_gama[["object_id", "ra", "dec", "z"]])
    quit()


