# Module SeaIceCalc
    # Input: NetCDF file include sea ice as an output
    # Output: Area weighted % ice concentration over the globe

import numpy as np
import xarray as xr


class SeaIceCalc:
    @staticmethod
    def ice_frac(ds, threshold=0.15, lat=slice(0,48)):
        """
        Compute area-weighted ice coverage on the globe
        """

        ds = ds.isel(lat=lat)

        # Extract sea ice concentration
        sic = ds["sic"]
        sic_mean = sic.mean(dim="time")
        lat = sic["lat"]

        # Extract weights
        weights = np.cos(np.deg2rad(lat))
        weights = weights.where(np.isfinite(weights), 0)
        weights_2d, _ = xr.broadcast(weights, sic_mean)

        # Mask ice to be over threshold to count towards ice coverage
        ice = sic_mean > threshold

        # Compute fraction
        num = (ice * weights_2d).sum(dim=["lat", "lon"])
        den = weights_2d.sum(dim=["lat", "lon"])
        ice_fraction = num / den * 100
        ice_fraction = ice_fraction.item()
        
        return ice_fraction

class Habitability:
    @staticmethod
    def fraction(ds, var, lat=slice(0,48)):
        """
        Compute area-weighted habitability (%) for:
        var = "sm", "st", or "tsurf"
        """

        ds = ds.isel(lat=lat)

        # Land mask
        sm = ds["sm"].isel(soil_layer=0)
        sm_mean = sm.mean("time")
        land = sm_mean > 1e-6

        # Variable extraction
        if var == "sm":
            data = ds["sm"].isel(soil_layer=0)    
            habitable = (sm_mean >= 0.1) & land
        
        elif var == "st":
            data = ds["st"].isel(soil_layer=0)
            mean = data.mean("time")
            habitable = (mean >= 0) & (mean <= 48) & land

        elif var == "tsurf":
            data = ds["tsurf"]
            mean = data.mean("time")
            habitable = (mean >= 0) & (mean <= 48) & land

        else:
            raise ValueError("var must be 'sm', 'st', or 'tsurf'")
        
        # Weights
        lat = data.lat
        weights = np.cos(np.deg2rad(lat))
        weights = weights.where(np.isfinite(weights), 0)
        weights_2d, _ = xr.broadcast(weights, land)

        # Area weighted fraction
        numerator = (habitable * weights_2d).sum(dim=["lat", "lon"])
        denominator = (land * weights_2d).sum(dim=["lat", "lon"])
        frac = ( numerator / denominator ) * 100

        return float(frac.item())
