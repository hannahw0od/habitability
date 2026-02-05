# Module SeaIceCalc
    # Input: NetCDF file include sea ice as an output
    # Output: Area weighted % ice concentration over the globe

import numpy as np
import xarray as xr


class SeaIceCalc:
    @staticmethod
    def ice_frac(ds, threshold=0.15, lat=slice(0,48)):
        """
        Compute area-weighted sea-ice coverage over the oceans
        """
        ds = ds.isel(lat=lat)

        # Sea ice concentration over the Ocean
        sic = ds["sic"].mean("time")

        # Mask ice to be over threshold to count towards ice coverage
        ice_present = sic >= threshold

        # Ocean fraction
        ocean_frac = 1.0 - ds["lsm"]

        # Latitude weights
        lat = sic["lat"]
        weights = np.cos(np.deg2rad(lat))
        weights = weights.where(np.isfinite(weights), 0)
        weights_2d, _ = xr.broadcast(weights, sic)


        # Numerator: ice area over the ocean
        num = (ice_present * ocean_frac * weights_2d).sum(dim=["lat", "lon"])

        # Denominator: total ocean area
        den = (ocean_frac * weights_2d).sum(dim=["lat", "lon"])

        # guard against zero ocean area
        if float(den) == 0.0:
            return 0.0

        ice_fraction = num / den * 100
        ice_fraction = ice_fraction.item()

        plot = sic * ocean_frac
        
        return ice_fraction, plot
    
class SST:
    @staticmethod
    def temp(ds,lat=slice(0,48)):
        """
        Compute area-weighted sea surface temperature over the oceans
        """
        ds = ds.isel(lat=lat)

        # Sea surface temperature for each grid cell averaged over time
        sst = ds["sst"].mean("time")

        # Ocean fraction
        ocean_frac = 1.0 - ds["lsm"]

        # Latitude weights
        lat = sst["lat"]
        weights = np.cos(np.deg2rad(lat))
        weights = weights.where(np.isfinite(weights), 0)
        weights_2d, _ = xr.broadcast(weights, sst)

        # Numerator: average sea surface temperature over the whole ocean
        num = (sst * ocean_frac * weights_2d).sum(dim=["lat", "lon"])

        # Denominator: total ocean area
        den = (ocean_frac * weights_2d).sum(dim=["lat", "lon"])

        # guard against zero ocean area
        if float(den) == 0.0:
            return 0.0

        return float((num / den).item())


class Habitability:
    @staticmethod
    def fraction(ds, var, lat=slice(0,48)):
        """
        Compute area-weighted habitability (%) for:
        var = "sm", "st", or "tsurf"
        """

        ds = ds.isel(lat=lat)

        # Land mask
        lsm = ds["lsm"]
        land = lsm > 0
        # sm = ds["sm"].isel(soil_layer=0)
        # sm_mean = sm.mean("time")
        # land = sm_mean > 1e-6

        # Variable extraction
        if var == "sm":
            data = ds["sm"].isel(soil_layer=0) 
            mean = data.mean("time")   
            habitable = (mean >= 0.1) & land
        
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

        return float(frac.item()), mean, habitable

