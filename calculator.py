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
        lsm = ds["lsm"]
        if "time" in lsm.dims:
            land = (lsm > 0).any("time")
        else:
            land = lsm > 0
        ocean_frac = 1.0 - land

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
        lsm = ds["lsm"]
        if "time" in lsm.dims:
            land = (lsm > 0).any("time")
        else:
            land = lsm > 0
        ocean_frac = 1.0 - land

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

        if "time" in lsm.dims:
            land = (lsm > 0).any("time")
        else:
            land = lsm > 0

        # land = lsm > 0
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
        weights_2d, _ = xr.broadcast(weights, mean)

        # Area weighted fraction
        numerator = (habitable * weights_2d).sum(dim=["lat", "lon"])
        denominator = (land * weights_2d).sum(dim=["lat", "lon"])
        frac = ( numerator / denominator ) * 100
        return float(frac.item()), mean, habitable

class SRD:
    @staticmethod
    def srd_calc(ds,lat=slice(0,48), grid="all"):
        """
        Compute area-weighted incoming shortwave radiation over all grid cells
        """
        ds = ds.isel(lat=lat)

        # Incoming shortwave radiation for each grid cell averaged over time
        srd = ds["srd"].mean("time")

        # Latitude weights
        lat = srd["lat"]
        weights = np.cos(np.deg2rad(lat))
        weights = weights.where(np.isfinite(weights), 0)
        weights_2d, _ = xr.broadcast(weights, srd)

        if grid == "all":
            # Numerator: average incoming shortwave radiation over all grid cells
            num = (srd * weights_2d).sum(dim=["lat", "lon"])

            # Denominator: total area
            den = (weights_2d).sum(dim=["lat", "lon"])
            # guard against zero area
            if float(den) == 0.0:
                return 0.0

            plot = srd

            return float((num / den).item()), plot
        
        elif grid == "land":
            # Numerator: average incoming shortwave radiation over land grid cells
            lsm = ds["lsm"]
            if "time" in lsm.dims:
                land = (lsm > 0).any("time")
            else:
                land = lsm > 0

            num = (srd * weights_2d * land).sum(dim=["lat", "lon"])

            # Denominator: total land area
            den = (weights_2d * land).sum(dim=["lat", "lon"])
            # guard against zero area
            if float(den) == 0.0:
                return 0.0
            
            plot = srd * land

            return float((num / den).item()), plot
    
        elif grid == "ocean":
            # Numerator: average incoming shortwave radiation over ocean grid cells
            lsm = ds["lsm"]
            if "time" in lsm.dims:
                land = (lsm > 0).any("time")
            else:
                land = lsm > 0

            ocean = 1.0 - land
            num = (srd * weights_2d * ocean).sum(dim=["lat", "lon"])

            # Denominator: total ocean area
            den = (weights_2d * ocean).sum(dim=["lat", "lon"])
            # guard against zero area
            if float(den) == 0.0:
                return 0.0
            
            plot = srd * ocean

            return float((num / den).item()), plot
        
        else:
            raise ValueError("grid must be 'all', 'land', or 'ocean'")

        
class Tempdiff:
    @staticmethod
    def temp_diff(ds, var, lat_slice):
        """
        Compute temperature difference between equator and either north or south pole for:
        var = "st" or "tsurf"
        """

        ds = ds.isel(lat=lat_slice)

        # Land mask
        lsm = ds["lsm"]
        if "time" in lsm.dims:
            land = (lsm > 0).any("time")
        else:
            land = lsm > 0

        # Variable extraction
        if var == "st":
            data = ds["st"].isel(soil_layer=0)
        
        elif var == "tsurf":
            data = ds["tsurf"]
        
        else:
            raise ValueError("var must be 'st' or 'tsurf'")

        # Masked time mean
        mean = data.mean("time")
        masked_mean = mean.where(land)

        # Weights
        latitude = masked_mean["lat"]
        weights = np.cos(np.deg2rad(latitude))
        weights = weights.where(np.isfinite(weights), 0)

        # Function to compute area-weighted mean over a boolean latitude mask
        def area_weighted_mean(masked_data,lat_mask):
            # Latitude mask for the region of interest
            latitudes = weights.where(lat_mask, 0)
            latitudes_2d, _ = xr.broadcast(latitudes, masked_data)
            num = (masked_data * latitudes_2d).sum(dim=["lat", "lon"])
            den = latitudes_2d.sum(dim=["lat", "lon"])
            return num / den if den != 0 else 0
        
        # Create latitude masks for equator and poles
        # Equator mask: latitude between -10 and 10 degrees
        equator_mask = (latitude >= -10) & (latitude <= 10)
        equator_temp = area_weighted_mean(masked_mean, equator_mask)

        # Polar masks
        north_pole_mask = latitude >= 80
        south_pole_mask = latitude <= -80

        # Calculate temperature depending on what user wants
        if lat_slice == slice(0,48):
            # Both poles
            polar_mask = north_pole_mask | south_pole_mask
        elif lat_slice == slice(0,24):
            # North pole only
            polar_mask = north_pole_mask
        elif lat_slice == slice(24,48):
            # South pole only
            polar_mask = south_pole_mask
        else:
            raise ValueError("lat slice must be either 0,48, 0,24, or 24,48")
        
        polar_temp = area_weighted_mean(masked_mean, polar_mask)

        return float((equator_temp - polar_temp).item()), float(equator_temp), float(polar_temp)

class Tempdiffall:
    @staticmethod
    def temp_diff(ds, lat_slice):
        """
        Compute temperature difference between equator and either north or south pole for:
        tsurf over all grid cells (no land mask)
        """

        ds = ds.isel(lat=lat_slice)

        # Masked time mean
        data = ds["tsurf"]
        mean = data.mean("time")

        # Weights
        latitude = mean["lat"]
        weights = np.cos(np.deg2rad(latitude))
        weights = weights.where(np.isfinite(weights), 0)

        # Function to compute area-weighted mean over a boolean latitude mask
        def area_weighted_mean(masked_data,lat_mask):
            # Latitude mask for the region of interest
            latitudes = weights.where(lat_mask, 0)
            latitudes_2d, _ = xr.broadcast(latitudes, masked_data)
            num = (masked_data * latitudes_2d).sum(dim=["lat", "lon"])
            den = latitudes_2d.sum(dim=["lat", "lon"])
            return num / den if den != 0 else 0
        
        # Create latitude masks for equator and poles
        # Equator mask: latitude between -10 and 10 degrees
        equator_mask = (latitude >= -10) & (latitude <= 10)
        equator_temp = area_weighted_mean(mean, equator_mask)

        # Polar masks
        north_pole_mask = latitude >= 80
        south_pole_mask = latitude <= -80

        # Calculate temperature depending on what user wants
        if lat_slice == slice(0,48):
            # Both poles
            polar_mask = north_pole_mask | south_pole_mask
        elif lat_slice == slice(0,24):
            # North pole only
            polar_mask = north_pole_mask
        elif lat_slice == slice(24,48):
            # South pole only
            polar_mask = south_pole_mask
        else:
            raise ValueError("lat slice must be either 0,48, 0,24, or 24,48")
        
        polar_temp = area_weighted_mean(mean, polar_mask)

        return float((equator_temp - polar_temp).item()), float(equator_temp), float(polar_temp)
