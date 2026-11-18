"""Dependency-light geographic helpers. All returned distances are metres."""
from __future__ import annotations
import numpy as np
EARTH_RADIUS_M = 6_371_008.8
def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 2*EARTH_RADIUS_M*np.arcsin(np.sqrt(np.clip(a, 0, 1)))
def bearing_deg(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2)); dlon=lon2-lon1
    return (np.degrees(np.arctan2(np.sin(dlon)*np.cos(lat2), np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(dlon)))+360)%360
def angular_difference_deg(a,b): return np.abs((np.asarray(a)-np.asarray(b)+180)%360-180)
class LocalProjector:
    """Local equirectangular projection, accurate for the trip-scale networks."""
    def __init__(self,lon,lat): self.lon0=float(np.nanmedian(lon)); self.lat0=float(np.nanmedian(lat)); self.cos_lat0=np.cos(np.radians(self.lat0))
    def transform(self,lon,lat):
        lon,lat=np.asarray(lon,dtype=float),np.asarray(lat,dtype=float)
        return np.radians(lon-self.lon0)*EARTH_RADIUS_M*self.cos_lat0,np.radians(lat-self.lat0)*EARTH_RADIUS_M
def local_projector(lon,lat): return LocalProjector(lon,lat)
def point_to_segments_m(points_xy,starts_xy,ends_xy):
    p,a,b=map(np.asarray,(points_xy,starts_xy,ends_xy)); ab=b-a; den=np.einsum('ij,ij->i',ab,ab)
    frac=np.divide(np.einsum('ij,ij->i',p-a,ab),den,out=np.zeros(len(p)),where=den>0); frac=np.clip(frac,0,1)
    return np.linalg.norm(p-(a+ab*frac[:,None]),axis=1),frac
