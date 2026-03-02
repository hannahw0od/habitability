import numpy as np
import xarray as xr


class SeaIce:
    @staticmethod
    def sic_timeseries(ds, lsm):

        # Mask ice to be over threshold to count towards ice coverage
        ice_present = ds >= 0.15

        # Ocean fraction
        ocean_frac = 1.0 - lsm

        # Latitude weights
        lat = ds["lat"]
        weights = np.cos(np.deg2rad(lat))
        weights = weights.where(np.isfinite(weights), 0)
        weights_2d, _ = xr.broadcast(weights, ds)


        # Numerator: ice area over the ocean
        num = (ice_present * ocean_frac * weights_2d).sum(dim=["lat", "lon"])

        # Denominator: total ocean area
        den = (ocean_frac * weights_2d).sum(dim=["lat", "lon"])

        # Avoid division by zero
        result = num / den.where(den != 0, np.nan)
        
        return result.values  # Return as numpy array

class WholeGlobeParam:
    def globalparam_timeseries(ds):
        param = ds

        # Latitude weights
        lat = param["lat"]
        weights = np.cos(np.deg2rad(lat))
        weights = weights.where(np.isfinite(weights), 0)
        weights_2d, _ = xr.broadcast(weights, param)

        # Numerator: parameter
        num = (param * weights_2d).sum(dim=["lat", "lon"])

        # Denominator: total area
        den = (weights_2d).sum(dim=["lat", "lon"])

        # Avoid division by zero
        result = num / den.where(den != 0, np.nan)
        
        return result.values  # Return as numpy array

class LandParam:
    def landparam_timeseries(ds, lsm):

        # Land mask
        land = lsm

        # Latitude weights
        lat = ds["lat"]
        weights = np.cos(np.deg2rad(lat))
        weights = weights.where(np.isfinite(weights), 0)
        weights_2d, _ = xr.broadcast(weights, ds)

        # Numerator: Parameter over land
        num = (ds * land * weights_2d).sum(dim=["lat", "lon"])

        # Denominator: total land area
        den = (land * weights_2d).sum(dim=["lat", "lon"])

        # Avoid division by zero
        result = num / den.where(den != 0, np.nan)
        
        return result.values  # Return as numpy array

class OceanParam:
    def oceanparam_timeseries(ds, lsm):

        # Land mask
        land = lsm
        ocean_frac = 1.0 - lsm

        # Latitude weights
        lat = ds["lat"]
        weights = np.cos(np.deg2rad(lat))
        weights = weights.where(np.isfinite(weights), 0)
        weights_2d, _ = xr.broadcast(weights, ds)

        # Numerator: Parameter over ocean
        num = (ds * ocean_frac * weights_2d).sum(dim=["lat", "lon"])

        # Denominator: total ocean area
        den = (ocean_frac * weights_2d).sum(dim=["lat", "lon"])

        # Avoid division by zero
        result = num / den.where(den != 0, np.nan)
        
        return result.values  # Return as numpy array
