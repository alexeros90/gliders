from dl import queryClient as qc, authClient as ac

def get_desi_sources(ra_range, dec_range, timeout = 60):
    # print(ac.whoAmI())
    sql = f"""
            SELECT TOP 15 zp.targetid, zp.mean_fiber_ra, zp.mean_fiber_dec,
                   zp.z, zp.spectype
            FROM desi_dr1.zpix AS zp
            WHERE (zp.survey='main') AND zp.main_primary
            AND zp.zwarn = 0 
            AND zp.mean_fiber_ra BETWEEN {ra_range[0]} AND {ra_range[1]}
            AND zp.mean_fiber_dec BETWEEN {dec_range[0]} AND {dec_range[1]}
            """
    qc.set_timeout_request(timeout)
    # print("DESI_DIR", qc.get_timeout_request())
    df = qc.query(sql=sql, fmt='pandas')
    df.rename(columns={"targetid": "object_id", "mean_fiber_ra": "ra", "mean_fiber_dec": "dec"}, inplace=True)
    df["object_id"] = df["object_id"].astype(str)
    return df

if __name__ == "__main__":
    df_desi = get_desi_sources([229.50, 229.54], [42.72, 42.76])
    print(df_desi[["object_id", "ra", "dec", "z", "spectype"]])
    quit()



# ALT QUERY
# sql = """
# SELECT TOP 10
#     zp.targetid,
#     zp.mean_fiber_ra,
#     zp.mean_fiber_dec,
#     p.ra AS target_ra,
#     p.dec AS target_dec,
#     zp.z,
#     zp.zwarn,
#     zp.spectype
# FROM desi_dr1.zpix AS zp
# JOIN desi_dr1.photometry AS p
#     ON zp.targetid = p.targetid
# WHERE zp.survey = 'main'
#   AND zp.main_primary
#   AND zp.random_id BETWEEN 0. AND 10.
# """

