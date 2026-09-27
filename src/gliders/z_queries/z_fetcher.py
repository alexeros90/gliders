import asyncio
import time
import pandas as pd

async def _query_nearby_z_sources_async(region_bounds, timeout=60, max_attempts=2, retry_wait=2):
    """
    Query all supported redshift catalogs concurrently.

    Parameters are equivalent to ``query_nearby_z_sources``.

    Returns
    -------
    tuple
        (combined_dataframe, failed_query), where ``failed_query`` is
        True if any catalog query failed.
    """

    from .DESI_Query import _get_desi_sources
    from .SDSS_Query import _get_sdss_sources
    from .NED_Query import _get_ned_sources
    from .GAMA_Query import _get_gama_sources

    failed_query = False

    tasks = {
        "DESI": _retry_async(lambda: asyncio.to_thread(_get_desi_sources, region_bounds[0], region_bounds[1], timeout),
            max_attempts, retry_wait),
        "GAMA": _retry_async(lambda: asyncio.to_thread(_get_gama_sources, region_bounds[0], region_bounds[1], timeout),
            max_attempts, retry_wait),
        "SDSS": _retry_async(lambda: asyncio.to_thread(_get_sdss_sources, region_bounds[0], region_bounds[1], 19, timeout),
            max_attempts, retry_wait),
        "NED": _retry_async(lambda: asyncio.to_thread(_get_ned_sources, region_bounds[0], region_bounds[1], True, timeout),
            max_attempts, retry_wait),
    }

    completed = await asyncio.gather(
        *tasks.values(),
        return_exceptions=True
    )

    results = {}
    for name, result in zip(tasks.keys(), completed):
        if isinstance(result, Exception):
            print(f"{name} failed: {result}")
            results[name] = None
            failed_query = True
        else:
            results[name] = result

    return _combine_query_results(results), failed_query

async def _retry_async(func, max_attempts, wait_seconds=2):
    """
    Execute an asynchronous callable with retry logic.

    Parameters
    ----------
    func : callable
        Awaitable callable to execute.
    max_attempts : int
        Maximum number of execution attempts.
    wait_seconds : float, optional
        Delay between retries.

    Raises
    ------
    Exception
        Re-raises the last exception after all attempts fail.
    """
    last_exception = None
    for attempt in range(max_attempts):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                await asyncio.sleep(wait_seconds)
    raise last_exception

def _retry_sync(func, max_attempts, wait_seconds=2):
    """
    Execute a callable with retry logic.

    Parameters
    ----------
    func : callable
        Function to execute.
    max_attempts : int
        Maximum number of execution attempts.
    wait_seconds : float, optional
        Delay between retries.

    Raises
    ------
    Exception
        Re-raises the last exception after all attempts fail.
    """
    last_exception = None
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                time.sleep(wait_seconds)
    raise last_exception

def _query_nearby_z_sources_sync(region_bounds, timeout=60, max_attempts=2, retry_wait=2):
    """
    Query all supported redshift catalogs sequentially.

    Parameters are equivalent to ``query_nearby_z_sources``.

    Returns
    -------
    tuple
        (combined_dataframe, failed_query), where ``failed_query`` is
        True if any catalog query failed.
    """

    from .DESI_Query import _get_desi_sources
    from .SDSS_Query import _get_sdss_sources
    from .NED_Query import _get_ned_sources
    from .GAMA_Query import _get_gama_sources

    failed_query = False

    results = {}

    query_functions = {
        "DESI": lambda: _get_desi_sources(region_bounds[0], region_bounds[1], timeout),
        "GAMA": lambda: _get_gama_sources(region_bounds[0], region_bounds[1], timeout),
        "SDSS": lambda: _get_sdss_sources(region_bounds[0], region_bounds[1], 19, timeout),
        "NED":  lambda: _get_ned_sources(region_bounds[0], region_bounds[1], True, timeout),
    }

    for name, func in query_functions.items():
        try:
            results[name] = _retry_sync(func, max_attempts, retry_wait)
        except Exception as e:
            print(f"{name} failed: {e}")
            results[name] = None
            failed_query = True

    return _combine_query_results(results), failed_query

def _combine_query_results(results_dict):
    """
    Merge catalog query results into a single cleaned DataFrame.

    Parameters
    ----------
    results_dict : dict
        Mapping between catalog names and query result DataFrames.

    Returns
    -------
    pandas.DataFrame
        Combined DataFrame containing the columns
        ``object_id``, ``ra``, ``dec`` and ``z``.
    """

    valid_dfs = [
        df[["object_id", "ra", "dec", "z"]]
        for df in results_dict.values()
        if df is not None and not df.empty
    ]

    if valid_dfs:
        all_z_sources = pd.concat(valid_dfs, ignore_index=True)
    else:
        all_z_sources = pd.DataFrame(
            columns=["object_id", "ra", "dec", "z"]
        )

    all_z_sources = all_z_sources.dropna(subset=["z"])
    all_z_sources["object_id"] = all_z_sources["object_id"].astype(str)

    return all_z_sources

def query_nearby_z_sources(region_bounds, timeout=60, synchronous=False, attempts=2):
    """
    Retrieve spectroscopic redshift sources within a sky region.

    This function queries all currently supported spectroscopic surveys
    (e.g. DESI, SDSS, NED and GAMA) and returns the available redshift
    measurements for objects located inside the specified region. Queries
    can be executed either concurrently (default) or sequentially.

    Parameters
    ----------
    region_bounds : tuple(tuple, tuple)
        Sky-region boundaries expressed as::

            ((ra_min, ra_max), (dec_min, dec_max))

        where right ascension and declination are given in degrees.

    timeout : float, optional
        Maximum time, in seconds, allowed for an individual catalog
        query before it is considered failed. Default is 60.

    synchronous : bool, optional
        If True, catalog queries are executed sequentially. If False
        (default), queries are executed concurrently using asyncio.

    attempts : int, optional
        Maximum number of attempts for each catalog query before giving
        up. Default is 2.

    Returns
    -------
    all_z_sources : pandas.DataFrame
        Table containing all successfully retrieved spectroscopic
        redshifts. The returned DataFrame includes the columns:

        - ``object_id`` : source identifier
        - ``ra`` : right ascension (degrees)
        - ``dec`` : declination (degrees)
        - ``z`` : spectroscopic redshift

    failed_query : bool
        True if one or more catalog queries failed after all retry
        attempts, False otherwise.

    Raises
    ------
    RuntimeError
        If called from an active asyncio event loop while
        ``synchronous=False``. In this case the asynchronous internal
        implementation should be awaited directly.

    Notes
    -----
    Failed catalog queries do not prevent successful results from other
    catalogs from being returned. The ``failed_query`` flag can be used
    to detect partial failures.
    """

    if synchronous:
        return _query_nearby_z_sources_sync(region_bounds, timeout, max_attempts = attempts, retry_wait = 2)
    else:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop → safe in scripts
            return asyncio.run(_query_nearby_z_sources_async(region_bounds, timeout, max_attempts = attempts, retry_wait = 2) )
        else:
            raise RuntimeError(
                "Active event loop detected. Use:\n\n"
                "    await query_nearby_z_sources_async(...)"
            )