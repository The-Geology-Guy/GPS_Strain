# Geo-Strain Calculator
# Project and method developed by Vincent Cronin (vince_cronin@baylor.edu)
# Python file created and maintained by Luke Pajer (luke.pajer@gmail.com)
# Last Updated: May 21, 2020

# Import Packages
import pandas as pd
import numpy as np
import math
from pylab import *
from scipy import linalg as la
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.transforms as mtrans
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib import patheffects

plt.rcParams['font.family'] = ["Georgia"]

class unavco_data:
    
    def __init__(self, **kwargs):
        self.start_time = kwargs.get('start_time', '')
        self.end_time = kwargs.get('end_time', '')
        
    def get_stations(self, minlon, maxlon, minlat, maxlat):
        # Returns a pandas dataframe with all sites within a specific set of coordinates
        import requests, io
        url_ = "https://web-services.unavco.org/gps/metadata/sites/v1?minlatitude="
        coordinates = str(minlat) + "&maxlatitude=" + str(maxlat) + "&minlongitude=" + str(minlon) + "&maxlongitude=" + str(maxlon)
        srt_ = "&starttime=" + str(self.start_time)
        end_ = "&endtime=" + str(self.end_time)
        full_url = url_ + coordinates + "&summary=true"
        urlData = requests.get(full_url).content
        rawData = pd.read_csv(io.StringIO(urlData.decode('utf-8')))
        return rawData

    def site_data(self, sites, **kwargs):
        # Generates a pandas dataframe with all of the site information
        period = kwargs.get('period', 365)
        
        from dateutil.relativedelta import relativedelta
        from dateutil.parser import parse
        
        data_ = []
        for i in range(3):
            location = sites[i]
            file = "https://web-services.unavco.org/gps/data/position/"
            start_ = "/v3?analysisCenter=cwu&referenceFrame=nam14&starttime=" + str(self.start_time)
            end_ = "&endtime=" + str(self.end_time)
            query = "&report=long&dataPostProcessing=Cleaned&refCoordOption=first_epoch"
            data_loop = pd.read_csv(file + location + start_ + end_ + query, skiprows=[i for i in range(0,8)])
            site = location
            lon = data_loop.head(1)[' E longitude'][0] - 360
            lat = data_loop.head(1)[' N latitude'][0]
            difference_in_years = relativedelta(parse(max(data_loop['Datetime'])), parse(min(data_loop['Datetime']))).years
            difference_in_years = difference_in_years if difference_in_years > 0 else 1
            e_vel = (data_loop[' delta E'].mean() / difference_in_years) * 1000
            e_unc = 0.01
            n_vel = (data_loop[' delta N'].mean() / difference_in_years) * 1000
            n_unc = 0.01
            data_.append(dict(zip(['site', 'longitude', 'latitude', 'E velocity (mm/yr)', 'E uncertainty (mm/yr)', 
                                   'N velocity (mm/yr)', 'N uncertainty (mm/yr)'], 
                                  [site, lon, lat, e_vel, e_unc, n_vel, n_unc])))
        data_df = pd.DataFrame(data_)
        return data_df

class strain_data:
    
    def __init__(self, data_unav):
        self.data_unav = data_unav
    
    class computation:
        pass
    
    def output_data(self, **kwargs):
        # read in computation for easier data retrieval
        computation = strain_data.computation()
        
        # Primary output for the sites
        data_df = self.data_unav
        pwr = kwargs.get('pwr', 7)
        
        # Convert to radians
        l_rads = data_df[['site', 'longitude', 'latitude']].copy()
        l_rads['longitude'] = l_rads['longitude'].apply(lambda x: x * (math.pi/180))
        l_rads['latitude'] = l_rads['latitude'].apply(lambda x: x * (math.pi/180))
        computation.l_rads = l_rads
        
        # Determine UTM Zone
        utm_z = data_df[['site', 'longitude']].copy()
        utm_z['UTM_Zone'] = utm_z['longitude'].apply(lambda x: (x + 180)/6)
        
        def utm_zone(x):
            if x - int(x) > 0:
                return int(x) + 1
            else:
                return int(x)
        
        utm_z['UTM_Zone'] = utm_z['UTM_Zone'].apply(lambda x: utm_zone(x))
        utm_z = utm_z[['site', 'UTM_Zone']]
        computation.utm_z = utm_z
        
        # Central Meridian of Zone (long0)
        cm_long0 = utm_z.copy()
        cm_long0['long0'] = cm_long0['UTM_Zone'].apply(lambda x: -183 + (6 * x))
        cm_long0 = cm_long0[['site', 'long0']]
        computation.cm_long0 = cm_long0
        
        # Central Meridian of Zone (long0) in radians
        cm_long0_r = cm_long0.copy()
        cm_long0_r['long0_r'] = cm_long0_r['long0'].apply(lambda x: x * math.pi/180)
        cm_long0_r = cm_long0_r[['site', 'long0_r']]
        computation.cm_long0_r = cm_long0_r
        
        # Central meridian of zone to the west (the 'pseudo' zone)
        def cm_west(x):
            if x == -177:
                return 177
            else:
                return x - 6
        p_z = cm_long0.copy()
        p_z['cm_pseudo_zone'] = cm_long0['long0'].apply(lambda x: cm_west(x))
        p_z = p_z[['site', 'cm_pseudo_zone']]
        computation.p_z = p_z
        
        # Central meridian of zone to the west (the 'pseudo' zone) in radians
        p_z_r = p_z.copy()
        p_z_r['cm_pseudo_zone_r'] = p_z_r['cm_pseudo_zone'].apply(lambda x: x * math.pi/180)
        p_z_r = p_z_r[['site', 'cm_pseudo_zone_r']]
        computation.p_z_r = p_z_r
        
        # UTM 'pseudo' zone
        utm_p_z = p_z.copy()
        utm_p_z['UTM_Pseudo_Zone'] = utm_p_z['cm_pseudo_zone'].apply(lambda x: (x + 180)/6)
        utm_p_z['UTM_Pseudo_Zone'] = utm_p_z['UTM_Pseudo_Zone'].apply(lambda x: utm_zone(x))
        computation.utm_p_z = utm_p_z
        
        # WGS84 datum
        a_wgs84 = 6378137
        b_wgs84 = 6356752.3142
        computation.a_wgs84 = a_wgs84
        computation.b_wgs84 = b_wgs84
        
        # Calculate key components
        k0 = 0.9996 
        computation.k0 = k0
        e = math.sqrt(1-b_wgs84**2/a_wgs84**2)
        computation.e = e
        e_2 = ((e * a_wgs84)/b_wgs84)**2
        computation.e_2 = e_2
        n = (a_wgs84 - b_wgs84)/(a_wgs84 + b_wgs84)
        computation.n = n
        
        # Calculate rho
        def calc_rho(x, e_, a_):
            p_1 = a_*(1-e_**2)
            p_2 = (1-(e_**2 * math.sin(x)**2))**(3/2)
            return p_1/p_2

        rho = l_rads[['site', 'latitude']].copy()
        rho['$\rho$'] = rho['latitude'].apply(lambda x: calc_rho(x, e, a_wgs84))
        rho = rho[['site', '$\rho$']]
        computation.rho = rho
        
        # Calculate nu
        def calc_nu(x, e_, a_):
            p_1 = a_
            p_2 = math.sqrt(1-(e_**2 * math.sin(x)**2))
            return p_1/p_2

        nu = l_rads[['site', 'latitude']].copy()
        nu['$\nu$'] = nu['latitude'].apply(lambda x: calc_nu(x, e, a_wgs84))
        nu = nu[['site', '$\nu$']]
        computation.nu = nu
        
        # Calculate p
        p_0 = l_rads[['site', 'longitude']].copy()
        p_1 = cm_long0_r.copy()
        p_merge = p_0.merge(p_1, on='site')
        p_merge['p'] = p_merge.longitude - p_merge.long0_r
        p = p_merge[['site', 'p']]
        computation.p = p
        
        # Calculate pseudo p
        p_p0 = l_rads[['site', 'longitude']].copy()
        p_p1 = p_z.copy()
        p_p2 = p_z_r.copy()
        p_p_m = p_p0.merge(p_p1, on='site').merge(p_p2, on='site')
        computation.p_p_m = p_p_m

        def pseudo_p(x, y, z):
            if y == 177:
                return abs(x) - z
            else:
                return x - z

        p_p_m['pseudo_p'] = p_p_m.apply(lambda x: pseudo_p(x.longitude, x.cm_pseudo_zone, x.cm_pseudo_zone_r), axis=1)
        pseudo_p = p_p_m[['site', 'pseudo_p']]
        computation.pseudo_p = pseudo_p
        
        # Matrix Components
        def mat_comps(x, e, m):
            if m == 'm1':
                return x*(1-((e**2)/4)-((3*(e**4))/64)-((5*(e**6))/256))
            elif m == 'm2':
                return math.sin(2*x)*(((3*(e**2))/8)+((3*(e**4))/32)+((45*(e**6))/1024))
            elif m == 'm3':
                return math.sin(4*x)*(((15*(e**4))/256)+((45*(e**6))/1024))
            else:
                return math.sin(6*x)*((35*(e**6))/3072)

        def m_comp(m): 
            m_ = l_rads[['site', 'latitude']].copy()
            i = 0
            while i < len(m):
                m_[m[i]] = m_['latitude'].apply(lambda x: mat_comps(x, e, m[i]))
                i+=1
            return m_

        m_comps = m_comp(m=['m1', 'm2', 'm3', 'm4'])
        computation.m_comps = m_comps
        
        # Calculate M
        def calc_M(x0, x1, x2, x3, a):
            eq_ = (x0 - x1 + x2 - x3)
            return a*eq_

        M = m_comps.copy()
        M['M'] = M.apply(lambda x: calc_M(x.m1, x.m2, x.m3, x.m4, a_wgs84), axis=1)
        M = M[['site', 'M']]
        computation.M = M
        
        # Calculate the K components
        def k_comps(x, M, nu, k0, e_2, k):
            if k == 'K1':
                return M*k0
            elif k == 'K2':
                return k0*nu*math.sin(2*x)/4
            elif k == 'K3':
                return (k0*nu*math.sin(x)*((math.cos(x))**3)/24)*(5-((math.tan(x))**2)+(9*e_2*((math.cos(x))**2))+(4*(e_2**2)*((math.cos(x))**4)))
            elif k == 'K4':
                return k0*nu*math.cos(x)
            else:
                return (k0*nu*((math.cos(x))**3)/6)*(1-((math.tan(x))**2)+(e_2*((math.cos(x))**2)))

        def k_comp(k): 
            k_0 = l_rads[['site', 'latitude']].copy()
            k_1 = M.copy()
            k_2 = nu.copy()
            k_ = k_0.merge(k_1, on='site').merge(k_2, on='site')
            i = 0
            while i < len(k):
                k_[k[i]] = k_.apply(lambda x: k_comps(x.latitude, x.M, x['$\nu$'], k0, e_2, k[i]), axis=1)
                i+=1
            k_ = k_[['site'] + k]
            return k_

        k_c = k_comp(k=['K1', 'K2', 'K3', 'K4', 'K5'])
        computation.k_c = k_c
        
        # True Northing and Easting
        def t_ne(K1, K2, K3, K4, K5, p, ne):
            if ne == 'northing':
                return K1+(K2*(p**2))+(K3*(p**4))
            else:
                return 500000+(K4*p)+(K5*(p**3))

        t_n_0 = k_c.merge(p, on='site')
        computation.t_n_0 = t_n_0

        t_n_0['true_northing'] = t_n_0.apply(lambda x: t_ne(x.K1, x.K2, x.K3, x.K4, x.K5, x.p, 'northing'), axis = 1)
        t_n_0['true_easting'] = t_n_0.apply(lambda x: t_ne(x.K1, x.K2, x.K3, x.K4, x.K5, x.p, 'easting'), axis = 1)
        t_n_e = t_n_0[['site', 'true_northing', 'true_easting']]
        computation.t_n_e = t_n_e
        
        # Pseudo Northing and Easting
        p_n_0 = k_c.merge(pseudo_p, on='site')
        p_n_0['pseudo_northing'] = p_n_0.apply(lambda x: t_ne(x.K1, x.K2, x.K3, x.K4, x.K5, x.pseudo_p, 'northing'), axis = 1)
        p_n_0['pseudo_easting'] = p_n_0.apply(lambda x: t_ne(x.K1, x.K2, x.K3, x.K4, x.K5, x.pseudo_p, 'easting'), axis = 1)
        p_n_e = p_n_0[['site', 'pseudo_northing', 'pseudo_easting']]
        computation.p_n_e = p_n_e
        
        # Westernmost Zone
        def w_z_():
            if np.std(utm_z.UTM_Zone) > 5:
                return 60
            else:
                return (np.sum(utm_z.UTM_Zone)/3)//1

        w_z = w_z_()
        w_z_avg = (np.sum(utm_z.UTM_Zone)/3)
        w_z_std = np.std(utm_z.UTM_Zone)
        computation.w_z = w_z
        computation.w_z_avg = w_z_avg
        computation.w_z_std = w_z_std
        
        # UTM coordinates relative to the westernmost zone, to be used in strain analysis
        def utm_w_z(x, w, t, p):
            if x == w:
                return t
            else:
                return p

        utm_0 = utm_z.copy()
        utm_1 = t_n_e.copy()
        utm_2 = p_n_e.copy()
        utm_w = utm_0.merge(utm_1, on='site').merge(utm_2, on='site')

        utm_w['UTM_w_z_easting'] = utm_w.apply(lambda x: utm_w_z(x.UTM_Zone, w_z, x.true_easting, x.pseudo_easting), axis=1)
        utm_w['UTM_w_z_northing'] = utm_w.apply(lambda x: utm_w_z(x.UTM_Zone, w_z, x.true_northing, x.pseudo_northing), axis=1)
        utm_w = utm_w[['site', 'UTM_w_z_easting', 'UTM_w_z_northing']]
        computation.utm_w = utm_w
        
        # Center of Triangle
        mean_n = utm_w.UTM_w_z_northing.mean()
        mean_e = utm_w.UTM_w_z_easting.mean()
        computation.mean_n = mean_n
        computation.mean_e = mean_e
        
        # Revised Locations
        sites_r = utm_w.copy()
        sites_r['revised_easting'] = sites_r['UTM_w_z_easting'].apply(lambda x: x - mean_e)
        sites_r['revised_northing'] = sites_r['UTM_w_z_northing'].apply(lambda x: x - mean_n)
        sites_r = sites_r[['site', 'revised_easting', 'revised_northing']]
        computation.sites_r = sites_r
        
        # Velocities converted from mm/yr to m/yr
        vel_m = self.data_unav.copy().drop(['longitude', 'latitude'], axis=1)
        vel_m['E velocity (m/yr)'] = vel_m['E velocity (mm/yr)'].apply(lambda x: x * 0.001)
        vel_m['E uncertainty (m/yr)'] = vel_m['E uncertainty (mm/yr)'].apply(lambda x: x * 0.001)
        vel_m['N velocity (m/yr)'] = vel_m['N velocity (mm/yr)'].apply(lambda x: x * 0.001)
        vel_m['N uncertainty (m/yr)'] = vel_m['N uncertainty (mm/yr)'].apply(lambda x: x * 0.001)
        vel_m = vel_m.drop(['E velocity (mm/yr)', 'E uncertainty (mm/yr)', 'N velocity (mm/yr)', 'N uncertainty (mm/yr)'], axis=1)
        computation.vel_m = vel_m
        
        # Matrix 1
        M1 = np.array([[sites_r.revised_easting], [sites_r.revised_northing]]).transpose()
        computation.M1 = M1
        
        # Matrix 2
        def mat2(x):
            mat_2 = pd.DataFrame()
            for i in range(3):
                x = sites_r.revised_easting[i]
                y = sites_r.revised_northing[i]
                list_s = [pd.Series(np.array([1, 0, (-1 * y), x, y, 0]).transpose()), pd.Series(np.array([0, 1, x, 0, x, y]).transpose())]
                mat_2 = mat_2.append(list_s, ignore_index=True)
                continue
            return mat_2

        M2 = mat2(x=sites_r)
        computation.M2 = np.array(M2)
        
        # Matrix 3
        M3 = la.inv(M2)
        M3 = pd.DataFrame(M3)
        computation.M3 = np.array(M3)
        
        # Matrix 4
        M4_ = pd.concat([vel_m['E velocity (m/yr)'], vel_m['N velocity (m/yr)']]).sort_index()
        M4 = np.array(M4_)[np.newaxis].T
        computation.M4 = M4
        
        # Matrix 5
        M5 = np.matrix(M3).dot(np.matrix(M4))
        computation.M5 = M5
        
        # North Unit Vector
        n_v_unit = [0, 1]
        computation.n_v_unit = n_v_unit
        
        # Translation Vector
        t_v = [float(M5[0]), float(M5[1])]
        computation.t_v = t_v
        
        # Magnitude of translation vector, or speed (m/yr)
        t_v_s = np.sqrt((t_v[0]**2)+(t_v[1]**2))
        computation.t_v_s = t_v_s
        
        # Unit Translation Vector
        t_v_unit = [(t_v[0]/t_v_s), (t_v[1]/t_v_s)]
        computation.t_v_unit = t_v_unit
        
        # Angle between north vector and unit trans vector
        n_t_a = math.acos((t_v_unit[0]*n_v_unit[0])+(t_v_unit[1]*n_v_unit[1]))*(180/math.pi)
        computation.n_t_a = n_t_a
        
        # Azimuth of trans vect (degrees clockwise from north)
        def trans_azi(x, y):
            if x < 0:
                return 360 - y
            else:
                return y

        t_v_azi = trans_azi(t_v[0], n_t_a)
        computation.t_v_azi = t_v_azi
        
        # Matrix M6
        M6 = np.array([[M5[-3], M5[-2]], [M5[-2], M5[-1]]])
        computation.M6 = M6
        
        # Eigen System
        def eigen_s(x0, x1, x2, x3):
            ev_0 = x0 + x3
            ev_1 = 4 * x1 * x2
            ev_2 = (x0 - x3)**2
            ev_3 = np.sqrt(ev_1 + ev_2)
            ev_a = (ev_0 + ev_3) / 2
            ev_b = (ev_0 - ev_3) / 2
            eigen = [ev_a, ev_b]
            return eigen

        e_s = eigen_s(float(M6[0][0]), float(M6[0][1]), float(M6[1][0]), float(M6[1][1]))
        computation.e_s = e_s
        
        # Calculate e1 and e2
        def det_e(e_sys):
            if e_sys[0] > e_sys[1]:
                return [e_sys[0], e_sys[1]]
            else:
                return [e_sys[1], e_sys[0]]

        e1_2 = det_e(e_s)
        computation.e1_2 = e1_2

        # Calculate e1 and e2 unit eigenvectors
        def unit_eigen(x, y, z):
            x_c = 1/np.sqrt(1+((x-y)/z)**2)
            y_c = ((x-y)/z)/np.sqrt(1+((x-y)/z)**2)
            return [x_c, y_c]

        e1_unit = unit_eigen(e1_2[0], float(M6[0][0]), float(M6[0][1]))
        computation.e1_unit = e1_unit
        e2_unit = unit_eigen(e1_2[1], float(M6[0][0]), float(M6[0][1]))
        computation.e2_unit = e2_unit
        
        # Angle between north vector and e1/e2 unit eigenvectors (Degrees)
        def find_angle(w, x, y, z):
            return math.acos((w*x)+(y*z))*(180/math.pi)

        nv_e1 = find_angle(e1_unit[0], n_v_unit[0], e1_unit[1], n_v_unit[1])
        computation.nv_e1 = nv_e1
        nv_e2 = find_angle(e2_unit[0], n_v_unit[0], e2_unit[1], n_v_unit[1])
        computation.nv_e2 = nv_e2
        
        # Azimuth of e1/e2 unit eigenvectors
        def az_e(x, y):
            if x < 0:
                return 360 - y
            else:
                return y

        e1_azi = az_e(e1_unit[0], nv_e1)
        computation.e1_azi = e1_azi
        e2_azi = az_e(e2_unit[0], nv_e2)
        computation.e2_azi = e2_azi

        # Alternate Azimuth of e1/e2 unit eigenvectors
        def a_az_e(x):
            if x < 180:
                return x + 180
            else:
                return x - 180

        e1_azi_a = a_az_e(e1_azi)
        computation.e1_azi_a = e1_azi_a
        e2_azi_a = a_az_e(e2_azi)
        computation.e2_azi_a = e2_azi_a
        
        # Maximum infinitesimal shear strain
        mis_strain = 2 * np.sqrt(((float(M6[0][0]) - float(M6[1][1])) / 2)**2 + (float(M6[0][1])**2))
        computation.mis_strain = mis_strain
        
        # Area Strain
        a_strain = e1_2[0] + e1_2[1]
        computation.a_strain = a_strain
        
        # Invariants of the infinitesimal strain rate tensor
        inv_0 = a_strain
        computation.inv_0 = inv_0
        inv_1 = e1_2[0] * e1_2[1]
        computation.inv_1 = inv_1
        inv_2 = inv_1
        computation.inv_2 = inv_2
        
        # Matrix 7
        def m7(x, y):
            v = pd.concat([x, y]).sort_index()
            v = np.array(list(v.apply(lambda x: 1 / (x**2))))
            return np.diag(v)

        M7 = pd.DataFrame(m7(vel_m['E uncertainty (m/yr)'], vel_m['N uncertainty (m/yr)']))
        computation.M7 = np.array(M7)
    
        # Matrix 8
        M8 = M2.T
        computation.M8 = M8
        
        # Matrix (m9.1 = m7 dot m2)
        M9_1 = M7.dot(M2)
        computation.M9_1 = M9_1
        
        # Matrix (m9.2 = m8 dot m9.1)
        M9_2 = M8.dot(M9_1)
        computation.M9_2 = M9_2
        
        # Matrix 9
        M9 = la.inv(M9_2)
        computation.M9 = M9
        
        # Primary Data Output
        
        fields_ = ['E component ± uncert [m/yr]', 'N component ± uncert [m/yr]', 'Azimuth [degrees]', 
                   'Speed [m/yr]', 'Rotation ± uncertainty [degrees/yr]', 'Rotation ± uncertainty [nano-rad/yr]', 'Direction of rotation', 
                   'Max horizontal extension (e1H) [nano-strain]', 'Azimuth of S1H [degrees]', 'Min horizontal extension (e2H) [nano-strain]', 
                   'Azimuth of S2H [degrees]', 'Max shear strain [nano-strain]', 'Area strain [nano-strain]']

        data_1 = str(round(float(M5[0]), 4)) + ' $\pm$ ' + str(round(float(M9[0][0]), 12))
        data_2 = str(round(float(M5[1]), 4)) + ' $\pm$ ' + str(round(float(M9[1][1]), 12))
        data_3 = str(round(float(M5[2]) * (180 / math.pi), 10)) + ' $\pm$ ' + str(round(np.sqrt(float(M9[2][2])) * (180 / math.pi), 12))
        data_4 = str(round(float(M5[2]) * (10**9), 4)) + ' $\pm$ ' + str(round(np.sqrt(float(M9[2][2])) * (10**9), 4))
        data_5 = 'Clockwise' if (float(M5[2]) * (10**9)) < 0 else 'Anti-Clockwise'
        data_6 = str(round(float(e1_2[0]) * (10**9), 4))
        data_7 = str(round(e1_azi, 4)) + ' or ' + str(round(e1_azi_a, 4))
        data_8 = str(round(float(e1_2[1]) * (10**9), 4))
        data_9 = str(round(e2_azi, 4)) + ' or ' + str(round(e2_azi_a, 4))

        values_ = [data_1, data_2, str(round(t_v_azi, 4)), str(round(t_v_s, 4)), data_3, data_4, data_5, 
                   data_6, data_7, data_8, data_9, str(round(mis_strain*(10**9), 4)), str(round(a_strain*(10**9), 4))]

        primary = pd.DataFrame(values_, index=fields_)
        primary.columns = ['Translation Vector']
        computation.primary_data = primary
        
        # Calculate the strain ellipse
        stretch = np.array([[float(M5[3]), 0], [0, float(M5[5])]])
        computation.stretch = stretch
        
        shear = np.array([[0, float(M5[4])/2], [float(M5[4])/2, 0]])
        computation.shear = shear
        
        theta = float(M5[2]) * (180/math.pi)
        rotation = array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        computation.rotation_tensor = rotation
        
        S = (stretch + shear)
        computation.stretch_tensor = S
        R = rotation
        F_ = R.dot(S) * 10**6 + np.array([[1, 0], [0, 1]])

        F = dot(F_, F_.T)
        computation.deformation_matrix = F

        B = F @ F.T
        C = F.T @ F

        V = la.sqrtm(B)
        computation.left_stretch_tensor = V
        U = la.sqrtm(C)
        computation.right_stretch_tensor = U
        
        R_r = la.inv(V) @ F
        R_l = F @ la.inv(U)
        
        return computation
        
class strain_viz:
    # Some of the Python Functions are adaptations of Ondrej Lexa's GitHub repository
    
    def __init__(self, strain_data):
        self.strain_data = strain_data
    
    def def_ellipse(self, V):
        # Draw strain ellipse from deformation gradient
        theta = linspace(0, 2*pi, 180)
        xc, yc = cos(theta), sin(theta)
        x,y = dot(V, [xc,yc])
        plt.plot(xc, yc, 'slategrey', x, y, lw=2, linestyle='--')
        plt.fill(xc, yc, 'w', alpha=0.45)
        u, s, v = svd(V)
        plt.plot(x, y, 'k', lw=2, zorder=40)
        plt.quiver(zeros(2), zeros(2),
                   hstack((s*u[0],-s*u[0])), hstack((s*u[1],-s*u[1])),
                   scale=1, units='xy', color=['tomato', 'cornflowerblue'], 
                   width=0.065, headaxislength=0, headlength=0, zorder=30)
        plt.quiver(zeros(2), zeros(2),
                   hstack((1,0)), hstack((0,1)),
                   scale=1, units='xy', color=['tomato', 'cornflowerblue'], 
                   width=0.065, linestyle='dashed', alpha=0.25, headaxislength=0, headlength=0, zorder=10)
        plt.quiver(zeros(2), zeros(2),
                   hstack((-1,0)), hstack((0,-1)),
                   scale=1, units='xy', color=['tomato', 'cornflowerblue'], 
                   width=0.065, linestyle='dashed', alpha=0.25, headaxislength=0, headlength=0, zorder=10)
        axis('equal')
        axis('off')

    def def_field(self, V, **kwargs):
        # Visualize displacement field from
        # displacement gradient
        alpha_ = kwargs.get('alpha', '1')
        F = asarray(V)
        J = F - eye(2)
        X, Y = meshgrid(linspace(-3, 3, 21),
                        linspace(-2, 2, 17))
        u, v = tensordot(J, [X, Y], axes=1)
        plt.quiver(X, Y, u, v, angles='xy', color='black', alpha=alpha_)
        axis('off')
        
    def get_center(sites_):
        # Locate the center of the triangle
        lonc = sites_.longitude.sum() / 3
        latc = sites_.latitude.sum() /3
        if lonc < -180:
            lonc = lonc + 360
        elif lonc > 180:
            lonc = lonc - 360
        return lonc, latc
    
    def end_df(sites_):
        sites = sites_
        first_site = pd.DataFrame(sites.head(1))
        last_site = pd.DataFrame(sites.tail(1))
        end_sites = pd.concat([first_site, last_site]).reset_index(drop=True)
        return end_sites
        
    def ellipse_plot(self, **kwargs):
        sites = self.strain_data
        V = kwargs.get('V', 'off')
        ax = kwargs.get('ax', None)
        fig = kwargs.get('fig', None)

        end_sites = strain_viz.end_df(sites)
        lonc, latc = strain_viz.get_center(sites)

        # To shift the Strain Ellipse about the center
        shiftx = kwargs.get('shiftx',  0)
        shifty = kwargs.get('shifty',  0)

        # Pick tiler type (http://maps.stamen.com/)
        map_tile_type = kwargs.get('map_tile_type', 'terrain-background')
        tiler = cimgt.Stamen(map_tile_type)
        mercator = tiler.crs

        # Figure Size
        if ax is None:
            # To shift the Strain Ellipse about the center
            shiftx = kwargs.get('shiftx',  0)
            shifty = kwargs.get('shifty',  0)
            bound_ = kwargs.get('bounds', 0.5)
            figx = kwargs.get('figx', 15)
            figy = kwargs.get('figy', 15)
            fig = plt.figure(figsize=(figx, figy))
            ax = fig.add_subplot(1, 1, 1, projection=mercator)
            ax.set_extent([sites.longitude.max()+bound_, sites.longitude.min()-bound_, sites.latitude.min()-bound_, sites.latitude.max()+bound_], crs=ccrs.PlateCarree())

        # Tiler Size
        tiler_size = kwargs.get('tiler_size', 1)
        ax.add_image(tiler, tiler_size, interpolation='spline36')

        ax.set_aspect(1, 'datalim')
        ax.gridlines(draw_labels=True)

        plt.plot(sites.longitude, sites.latitude, color='blue', linestyle='--', linewidth=2, marker=',', transform=ccrs.PlateCarree(), zorder=20)
        plt.plot(end_sites.longitude, end_sites.latitude, color='blue', linestyle='--', linewidth=2, marker=',', transform=ccrs.PlateCarree(), zorder=20)
        plt.plot(sites.longitude, sites.latitude, color='black', linewidth=0, marker=',', transform=ccrs.PlateCarree(), label=sites.site, zorder=20)

        bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        my_dpi = fig.dpi
        
        length = kwargs.get('length', 25)
        scale_loc = kwargs.get('scale_loc', (0.5, 0.05))

        llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())

        sbllx = (llx1 + llx0) / 2
        sblly = lly0 + (lly1 - lly0) * scale_loc[1]

        tmc = ccrs.TransverseMercator(sbllx, sblly)

        x0, x1, y0, y1 = ax.get_extent(tmc)

        sbx = x0 + (x1 - x0) * scale_loc[0]
        sby = y0 + (y1 - y0) * scale_loc[1]
        # print(sbx, sby)

        sbxe = ((sbx + length * 500)/5)*2

        sbxf = round(sbx - length * 500)

        j = sbxf
        k = 1
        while k <= 5:
            bar_xs = [j, j + sbxe]
            if k % 2 == 0:
                ax.plot(bar_xs, [sby, sby], transform=tmc, solid_capstyle='butt', color='w', linewidth=15, zorder=10)
            else:
                ax.plot(bar_xs, [sby, sby], transform=tmc, solid_capstyle='butt', color='k', linewidth=15, zorder=11)
            j += sbxe
            k += 1
        
        buffer = [patheffects.withStroke(linewidth=1.5, foreground="w")]
        
        hei_ = kwargs.get('hei_', 5)
        
        ax.text(-1*sbxf, sby+(hei_*sby), str(length) + ' km', transform=tmc, fontsize=12,
                family='Arial', path_effects=buffer, horizontalalignment='left', verticalalignment='bottom')

        ax.text(sbxf, sby+(hei_*sby), '0 km', transform=tmc, fontsize=12,
                family='Arial', path_effects=buffer, horizontalalignment='right', verticalalignment='bottom')

        # Add Colors to site locations
        color_list = kwargs.get('color_list', ['g', 'b', 'r'])
        arrows = kwargs.get('arrows', 'show')
        
        for i in range(len(sites)):
            plt.draw()
            lon, lat = sites.longitude[i], sites.latitude[i]
            trans = ccrs.PlateCarree()._as_mpl_transform(ax)
            x, y = trans.transform_point((lon, lat))
            x_ = ((x/my_dpi))/width
            y_ = ((y/my_dpi))/height
            axi = fig.add_axes([(x_ - (5/width)*0.5), (y_ - (5/height)*0.5), (5/width), (5/height)])    
            colors = color_list
            scale_arrow = kwargs.get('scale_arrow', 40)
            if arrows == 'show':
                axi.quiver(sites['E velocity (mm/yr)'][i], sites['N velocity (mm/yr)'][i], scale=scale_arrow, width=0.0175, headwidth=3.5, color='k')
            axi.plot(0, 0, marker='o', markersize=10, color=colors[i])
            axi.axis('equal')
            axi.axis('off')

        sites_h = []
        for i in range(3):
            site_0 = Line2D([0], [0], marker='o', color='b', linestyle='--',fillstyle='full', markeredgecolor='red',
                            markeredgewidth=0.0, label=sites.site[i], markerfacecolor=color_list[i], markersize=15)
            sites_h.append(site_0)
            
        # Set Legend Location
        loc_ = kwargs.get('loc', 'upper center')
        
        # Add Legend
        leg = ax.legend(handles=[sites_h[0], sites_h[1], sites_h[2]], ncol=3, loc=loc_, fontsize="x-large")
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(0.5)
        leg.get_frame().set_alpha(0.75)
        
        # Add Strain Ellipse
        if V is not 'off':
            plt.draw()
            lon, lat = lonc, latc
            trans = ccrs.PlateCarree()._as_mpl_transform(ax)
            x, y = trans.transform_point((lon, lat))

            x_ = ((x/my_dpi))/width
            y_ = ((y/my_dpi))/height

            ax2 = fig.add_axes([(x_), (y_), 0.2, 0.2])
            ax2.set_xlim([-1,1])
            ax2.set_ylim([-1,1])
            strain_viz.def_ellipse(self, V)
            ax2.axis('equal')
            ax2.axis('off')
            p1 = ax.get_position()
            p2 = ax2.get_position()
            ax2.set_position([x_ - (p2.width/2 + shiftx), y_ - (p2.height/2 + shifty), p2.width, p2.height])
 
        axn = fig.add_axes([(x_), (y_), 0.05, 0.05])
        buffer = [patheffects.withStroke(linewidth=4, foreground="w")]
        axn.text(0.5, 0.0,u'\u25B2 \nN ', ha='center', fontsize=35, family='Arial', path_effects=buffer, rotation = 0)
        axn.axis('equal')
        axn.axis('off')
        p3 = ax.get_position()
        p4 = axn.get_position()
        axn.set_position([p3.x0 + (0.05*p3.x1), p3.y0 + (0.05*p3.y1), 0.05, 0.05])
        
        save_fig = kwargs.get('save_fig', None)
        
        if save_fig is not None:
            plt.savefig(str(save_fig), edgecolor='k', bbox_inches='tight')

    def symbol_map(self, **kwargs):
        sites = self.strain_data
        
        ax = kwargs.get('ax', None)
        fig = kwargs.get('fig', None)

        end_sites = strain_viz.end_df(sites)
        lonc, latc = strain_viz.get_center(sites)

        # To shift the Strain Ellipse about the center
        shiftx = kwargs.get('shiftx',  0)
        shifty = kwargs.get('shifty',  0)

        # Pick tiler type (http://maps.stamen.com/)
        map_tile_type = kwargs.get('map_tile_type', 'terrain-background')
        tiler = cimgt.Stamen(map_tile_type)
        mercator = tiler.crs

        if ax is None:
            # To shift the Strain Ellipse about the center
            shiftx = kwargs.get('shiftx',  0)
            shifty = kwargs.get('shifty',  0)
            bound_ = kwargs.get('bounds', 0.5)
            figx = kwargs.get('figx', 15)
            figy = kwargs.get('figy', 15)
            fig = plt.figure(figsize=(figx, figy))
            ax = fig.add_subplot(1, 1, 1, projection=mercator)
            ax.set_extent([sites.longitude.max()+bound_, sites.longitude.min()-bound_, sites.latitude.min()-bound_, sites.latitude.max()+bound_], crs=ccrs.PlateCarree())

        # Tiler Size
        tiler_size = kwargs.get('tiler_size', 1)
        ax.add_image(tiler, tiler_size, interpolation='spline36')

        ax.set_aspect(1, 'datalim')
        ax.gridlines(draw_labels=True)

        plt.plot(sites.longitude, sites.latitude, color='blue', linestyle='--', linewidth=2, marker=',', transform=ccrs.PlateCarree(), zorder=20)
        plt.plot(end_sites.longitude, end_sites.latitude, color='blue', linestyle='--', linewidth=2, marker=',', transform=ccrs.PlateCarree(), zorder=20)
        plt.plot(sites.longitude, sites.latitude, color='black', linewidth=0, marker=',', transform=ccrs.PlateCarree(), label=sites.site, zorder=20)

        bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        my_dpi = fig.dpi
        
        length = kwargs.get('length', 25)
        scale_loc = kwargs.get('scale_loc', (0.5, 0.05))

        llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())

        sbllx = (llx1 + llx0) / 2
        sblly = lly0 + (lly1 - lly0) * scale_loc[1]

        tmc = ccrs.TransverseMercator(sbllx, sblly)

        x0, x1, y0, y1 = ax.get_extent(tmc)

        sbx = x0 + (x1 - x0) * scale_loc[0]
        sby = y0 + (y1 - y0) * scale_loc[1]

        sbxe = ((sbx + length * 500)/5)*2

        sbxf = round(sbx - length * 500)

        j = sbxf
        k = 1
        while k <= 5:
            bar_xs = [j, j + sbxe]
            if k % 2 == 0:
                ax.plot(bar_xs, [sby, sby], transform=tmc, solid_capstyle='butt', color='w', linewidth=15, zorder=10)
            else:
                ax.plot(bar_xs, [sby, sby], transform=tmc, solid_capstyle='butt', color='k', linewidth=15, zorder=11)
            j += sbxe
            k += 1
        
        buffer = [patheffects.withStroke(linewidth=2.5, foreground="w")]
        
        hei_ = kwargs.get('hei_', 5)
        
        ax.text(-1*sbxf, sby+(hei_*sby), str(length) + ' km', transform=tmc, fontsize=12,
                family='Arial', path_effects=buffer, horizontalalignment='left', verticalalignment='bottom')

        ax.text(sbxf, sby+(hei_*sby), '0 km', transform=tmc, fontsize=12,
                family='Arial', path_effects=buffer, horizontalalignment='right', verticalalignment='bottom')

        # Add Colors to site locations
        color_list = kwargs.get('color_list', ['g', 'b', 'r'])
        arrows = kwargs.get('arrows', 'off')
        
        for i in range(len(sites)):
            plt.draw()
            lon, lat = sites.longitude[i], sites.latitude[i]
            trans = ccrs.PlateCarree()._as_mpl_transform(ax)
            x, y = trans.transform_point((lon, lat))
            x_ = ((x/my_dpi))/width
            y_ = ((y/my_dpi))/height
            axi = fig.add_axes([(x_ - (5/width)*0.5), (y_ - (5/height)*0.5), (5/width), (5/height)])    
            colors = color_list
            scale_arrow = kwargs.get('scale_arrow', 40)
            if arrows == 'show':
                axi.quiver(sites['E velocity (mm/yr)'][i], sites['N velocity (mm/yr)'][i], scale=scale_arrow, width=0.0175, headwidth=3.5, color='k')
            axi.plot(0, 0, marker='o', markersize=10, color=colors[i])
            axi.axis('equal')
            axi.axis('off')

        sites_h = []
        for i in range(3):
            site_0 = Line2D([0], [0], marker='o', color='b', linestyle='--',fillstyle='full', markeredgecolor='red',
                            markeredgewidth=0.0, label=sites.site[i], markerfacecolor=color_list[i], markersize=15)
            sites_h.append(site_0)
            
        # Set Legend Location
        loc_ = kwargs.get('loc', 'upper center')
        
        # Add Legend
        leg = ax.legend(handles=[sites_h[0], sites_h[1], sites_h[2]], ncol=3, loc=loc_, fontsize="x-large")
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(0.5)
        leg.get_frame().set_alpha(0.75)
        
        plt.draw()
        
        # Add in the e1 and e2 symbols
        e1 = kwargs.get('e1', None)
        e2 = kwargs.get('e2', None)
        #e_loc = kwargs.get('e_loc', 'lower left')
        e_rot = kwargs.get('e_rot', 0)
        old_range = kwargs.get('old_range', [0.1, 300])
        new_range_a = kwargs.get('new_range_a', [40, 80])
        new_range_b = kwargs.get('new_range_b', [10, 15])
        max_strain = kwargs.get('max_strain', 300)
        min_strain = kwargs.get('min_strain', 0.1)
        
        # Add Map Symbol
        if None not in (e1, e2):
            plt.draw()
            lon, lat = lonc, latc
            trans = ccrs.PlateCarree()._as_mpl_transform(ax)
            x, y = trans.transform_point((lon, lat))

            x_ = ((x/my_dpi))/width
            y_ = ((y/my_dpi))/height

            ax2 = fig.add_axes([(x_), (y_), (5/width), (5/height)])
            ax2.set_xlim([-1,1])
            ax2.set_ylim([-1,1])
            strain_viz.map_symbol(self, e1, e2, rot=e_rot, old_range=old_range, new_range_a=new_range_a, new_range_b=new_range_b, max_strain=max_strain, min_strain=min_strain, ax=ax2)
            ax2.axis('equal')
            #ax2.axis('off')
            p1 = ax.get_position()
            p2 = ax2.get_position()
            ax2.set_position([x_ - (p2.width/2 + shiftx), y_ - (p2.height/2 + shifty), p2.width, p2.height])
            ax2.autoscale(False)
            
        plt.draw()
        
        axn = fig.add_axes([(x_), (y_), 0.05, 0.05])
        buffer = [patheffects.withStroke(linewidth=4, foreground="w")]
        axn.text(0.5, 0.0,u'\u25B2 \nN ', ha='center', fontsize=35, family='Arial', path_effects=buffer, rotation = 0)
        axn.axis('equal')
        axn.axis('off')
        p3 = ax.get_position()
        p4 = axn.get_position()
        axn.set_position([p3.x0 + (0.05*p3.x1), p3.y0 + (0.05*p3.y1), 0.05, 0.05])
        
        save_fig = kwargs.get('save_fig', None)
        
        if save_fig is not None:
            plt.savefig(str(save_fig), edgecolor='k', bbox_inches='tight')
            
    def scale_arrow(value, old_range, new_range):
        tmin, tmax = old_range
        xmin, xmax = new_range
        percent = abs((value - tmin) / (tmax - tmin))
        return ((xmax - xmin) * percent) + xmin

    def scale_arrow_percent(value, old_range):
        tmin, tmax = old_range
        return abs((value - tmin) / (tmax - tmin))

    def map_symbol(self, e1, e2, **kwargs):
        # Add Figure to plot        
        ax = kwargs.get('ax', 'none')
        rot = kwargs.get('rot', 0)
        old_range = kwargs.get('old_range', [0.1, 300])
        new_range_a = kwargs.get('new_range_a', [40, 80])
        new_range_b = kwargs.get('new_range_b', [10, 15])
        max_strain = kwargs.get('max_strain', 300)
        min_strain = kwargs.get('min_strain', 0.1)
        
        sz_e1 = strain_viz.scale_arrow(e1 * 10**9, old_range, new_range_a)
        sz_e2 = strain_viz.scale_arrow(e2 * 10**9, old_range, new_range_a)
        
        sz_e1_d = strain_viz.scale_arrow(e1 * 10**9, old_range, new_range_b)
        sz_e2_d = strain_viz.scale_arrow(e2 * 10**9, old_range, new_range_b)
        
        sz_p_e1 = strain_viz.scale_arrow(e1 * 10**9, [min_strain, max_strain], [0.2, 0.6])
        sz_p_e2 = strain_viz.scale_arrow(e2 * 10**9, [min_strain, max_strain], [0.2, 0.6])
        
        scale_arrow_percent_0 = strain_viz.scale_arrow(e1 * 10**9, [min_strain, max_strain], [0.2, 0.6])
        boxstyle0_d = f"darrow,pad=%s" % (scale_arrow_percent_0)

        scale_arrow_percent_1 = strain_viz.scale_arrow(e2 * 10**9, [min_strain, max_strain], [0.2, 0.6])
        boxstyle1_d = f"darrow,pad=%s" % (scale_arrow_percent_1)
        
        #scale_arrow_percent_1 = str(round(strain_viz.scale_arrow_percent(e2 * 10**9, old_range), 1))
        #boxstyle1_l = f"larrow,pad=%s" % (scale_arrow_percent_1)
        #boxstyle1_r = f"rarrow,pad=%s" % (scale_arrow_percent_1)
        
        if ax == 'none':
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(1, 1, 1)
        
        ax.spines['left'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('center')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')    
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])

        if (e1 == 0) and (e2 < 0):
            rot0 = mtrans.Affine2D().rotate_deg(rot)
            
            x0, y0 = rot0.transform_point((0.0, sz_p_e2))
            x1, y1 = rot0.transform_point((0.0, -sz_p_e2))
            
            ax.annotate("", 
                        xy=(0.0, 0.0),
                        xytext=(x0, y0), textcoords='data',
                        size=sz_e2, va="center", ha="center", color='k',
                        arrowprops=dict(arrowstyle="simple, head_length=0.35,head_width=0.5,tail_width=0.2", fc="k", ec='k', lw=2))


            ax.annotate("",
                        xy=(0.0,0.0),
                        xytext=(x1, y1),
                        size=sz_e2, va="center", ha="center", color='k',
                        arrowprops=dict(arrowstyle="simple, head_length=0.35,head_width=0.5,tail_width=0.2", fc="k", ec='k', lw=2))
            
        elif (e1 > 0) and (e2 == 0):
            bbox_props1 = dict(boxstyle=boxstyle0_d, fc="w", ec="k", lw=3)
            
            sz_text1 = "---------------" + ('-' * int(20*float(scale_arrow_percent_1)))
            
            ax.text(0.0, 0.0, sz_text1, ha="center", va="center", rotation=rot + 90,
                        size=sz_e1_d, color='w',
                        bbox=bbox_props1)
            
        elif (e1 > 0) and (e2 > 0):
            bbox_props2 = dict(boxstyle=boxstyle1_d, fc="w", ec="k", lw=3)
            
            sz_text1 = "---------------" + ('-' * int(20*float(scale_arrow_percent_1)))
            
            ax.text(0.0, 0.0, sz_text1, ha="center", va="center", rotation=rot,
                        size=sz_e2_d, color='w',
                        bbox=bbox_props2)
            
            sz_text0 = "---------------" + ('-' * int(20*float(scale_arrow_percent_0)))

            bbox_props3 = dict(boxstyle=boxstyle0_d, fc="w", ec="k", lw=3)
            ax.text(0.0, 0.0, sz_text0, ha="center", va="center", rotation=rot+90,
                        size=sz_e1_d, color='w',
                        bbox=bbox_props3)
            
        elif (e1 > 0) and (e2 < 0):
            angle_phi = rot

            l2 = np.array((5, 5))

            trans_angle = plt.gca().transData.transform_angles(np.array((angle_phi,)),
                                                               l2.reshape((1, 2)))[0]

            bbox_props = dict(boxstyle=boxstyle0_d, fc="w", ec="k", lw=3)
            
            sz_text = "---------------" + ('-' * int(20*float(scale_arrow_percent_0)))
            
            t = ax.text(0.0, 0.0, sz_text, ha="center", va="center",
                        size=sz_e1_d, color='w', rotation=trans_angle, bbox=bbox_props)

            rot1 = mtrans.Affine2D().rotate_deg(angle_phi)
            x0, y0 = rot1.transform_point((0.0, sz_p_e2))
            x1, y1 = rot1.transform_point((0.0, -sz_p_e2))

            ax.annotate("",
                        xy=(0.0, 0.0),
                        xytext=(x0, y0), textcoords='data',
                        size=sz_e2, va="center", ha="center", color='k',
                        arrowprops=dict(arrowstyle="simple, head_length=0.35,head_width=0.5,tail_width=0.2", fc="k", ec='k', lw=2))

            ax.annotate("",
                        xy=(0.0,0.0),
                        xytext=(x1, y1),
                        size=sz_e2, va="center", ha="center", color='k',
                        arrowprops=dict(arrowstyle="simple, head_length=0.35,head_width=0.5,tail_width=0.2", fc="k", ec='k', lw=2))
            
        elif (e1 < 0) and (e2 < 0):

            rot0 = mtrans.Affine2D().rotate_deg(rot)
            x0, y0 = rot0.transform_point((0.0, sz_p_e2))
            x1, y1 = rot0.transform_point((0.0, -sz_p_e2))
            x2, y2 = rot0.transform_point((sz_p_e1, 0.0))
            x3, y3 = rot0.transform_point((-sz_p_e1, 0.0))

            ax.annotate("",
                        xy=(0.0, 0.0),
                        xytext=(x0, y0), textcoords='data',
                        size=sz_e2, va="center", ha="center", color='k',
                        arrowprops=dict(arrowstyle="simple, head_length=0.35,head_width=0.5,tail_width=0.2", fc="k", ec='k', lw=2))

            ax.annotate("",
                        xy=(0.0,0.0),
                        xytext=(x1, y1),
                        size=sz_e2, va="center", ha="center", color='k',
                        arrowprops=dict(arrowstyle="simple, head_length=0.35,head_width=0.5,tail_width=0.2", fc="k", ec='k', lw=2))

            ax.annotate("",
                        xy=(0.0,0.0),
                        xytext=(x2, y2),
                        size=sz_e1, va="center", ha="center", color='k',
                        arrowprops=dict(arrowstyle="simple, head_length=0.35,head_width=0.5,tail_width=0.2", fc="k", ec='k', lw=2))

            ax.annotate("",
                        xy=(0.0,0.0),
                        xytext=(x3, y3),
                        size=sz_e1, va="center", ha="center", color='k',
                        arrowprops=dict(arrowstyle="simple, head_length=0.35,head_width=0.5,tail_width=0.2", fc="k", ec='k', lw=2))
        axis('off')
        
    def symbol_map_full(self, **kwargs):
        sites = self.strain_data
        V = kwargs.get('V', None)
        
        # Tiler Size
        tiler_size = kwargs.get('tiler_size', 1)
        
        # Add Colors to site locations
        color_list = kwargs.get('color_list', ['g', 'b', 'r'])
        arrows = kwargs.get('arrows', 'off')
        
        # Set Legend Location
        loc_ = kwargs.get('loc', 'upper center')
        
        # Get data for plot
        e1 = kwargs.get('e1', None)
        e2 = kwargs.get('e2', None)
        e_loc = kwargs.get('e_loc', 'lower left')
        e_rot = kwargs.get('e_rot', 0)
        
        # Import Site data and find center
        end_sites = strain_viz.end_df(sites)
        lonc, latc = strain_viz.get_center(sites)

        # To shift the Strain Ellipse about the center
        shiftx = kwargs.get('shiftx',  0)
        shifty = kwargs.get('shifty',  0)
        bound_ = kwargs.get('bounds', 0.5)

        # Pick tiler type (http://maps.stamen.com/)
        map_tile_type = kwargs.get('map_tile_type', 'terrain-background')
        tiler = cimgt.Stamen(map_tile_type)
        mercator = tiler.crs

        # Figure Size
        fig = plt.figure(figsize=(20, 15), constrained_layout=False)
        gs = gridspec.GridSpec(30, 40, figure=fig, wspace=0.0, hspace=0.0)
        ax = fig.add_subplot(gs[:, 11:], projection=mercator)
        ax.set_extent([sites.longitude.max()+bound_, sites.longitude.min()-bound_, sites.latitude.min()-bound_, sites.latitude.max()+bound_], crs=ccrs.PlateCarree())
        
        scale_arrow = kwargs.get('scale_arrow', 40)
        length = kwargs.get('length', 25)
        scale_loc = kwargs.get('scale_loc', (0.5, 0.05))
        old_range = kwargs.get('old_range', [0.1, 300])
        new_range_a = kwargs.get('new_range_a', [40, 80])
        new_range_b = kwargs.get('new_range_b', [10, 15])
        max_strain = kwargs.get('max_strain', 300)
        min_strain = kwargs.get('min_strain', 0.1)
        hei_ = kwargs.get('hei_', 5)
        map_tile_type = kwargs.get('map_tile_type', 'terrain-background')

        strain_viz.symbol_map(self, e1=e1, e2=e2, e_loc=e_loc, e_rot=e_rot, hei_=hei_, old_range=old_range, new_range_a=new_range_a, 
                              new_range_b=new_range_b, max_strain=max_strain, min_strain=min_strain, 
                              arrows=arrows, color_list=color_list, tiler_size=tiler_size, map_tile_type=map_tile_type,
                              scale_arrow=scale_arrow, length=length, scale_loc=scale_loc, loc_=loc_, ax=ax, fig=fig)

        ax1 = fig.add_subplot(gs[27:30, 1:9])
        
        image = kwargs.get('image', "https://www.unavco.org/education/resources/lib/images/unavco-logo-red-white-shadow.png")

        strain_viz.unavco_logo(image=image, ax=ax1)

        ax1_1 = fig.add_subplot(gs[:3, :10])
        
        title_ = kwargs.get('title', "GPS Triangle-Strain Map\nUsing UNAVCO PBO Data")
        fontsize_ = kwargs.get('fontsize', 24)
        ha_ = kwargs.get('ha', 'center')
        va_ = kwargs.get('va', 'top')
        xy_ = kwargs.get('xy', (0.5, 0.925))
        
        strain_viz.map_title(title=str(title_), xy=xy_, fontsize=fontsize_, ha=ha_, va=va_, ax=ax1_1)

        ax2 = fig.add_subplot(gs[4:12, 1:9])

        strain_viz.ellipse_subplot(self, V=V, ax=ax2)

        ax3 = fig.add_subplot(gs[13:18, :10])
        
        max_strain = kwargs.get('max_strain', 300)
        min_strain = kwargs.get('min_strain', 0.1)
        old_range = kwargs.get('old_range', [0.1, 300])

        strain_viz.contraction(old_range=old_range, max_strain=max_strain, min_strain=min_strain, ax=ax3)

        ax4 = fig.add_subplot(gs[20:25, :10])

        strain_viz.elongation(old_range=old_range, max_strain=max_strain, min_strain=min_strain, ax=ax4)
        
        save_fig = kwargs.get('save_fig', None)
        
        if save_fig is not None:
            plt.savefig(str(save_fig), edgecolor='k', bbox_inches='tight')
        
    def strain_map_full(self, **kwargs):
        sites = self.strain_data
        V = kwargs.get('V', None)
        
        # Tiler Size
        tiler_size = kwargs.get('tiler_size', 1)
        
        # Add Colors to site locations
        color_list = kwargs.get('color_list', ['g', 'b', 'r'])
        arrows = kwargs.get('arrows', 'show')
        size = kwargs.get('size', 10)
        label = kwargs.get('label', '10 mm/yr')
        
        # Set Legend Location
        loc_ = kwargs.get('loc', 'upper center')
        
        # Import Site data and find center
        end_sites = strain_viz.end_df(sites)
        lonc, latc = strain_viz.get_center(sites)

        # To shift the Strain Ellipse about the center
        shiftx = kwargs.get('shiftx',  0)
        shifty = kwargs.get('shifty',  0)
        bound_ = kwargs.get('bounds', 0.5)

        # Pick tiler type (http://maps.stamen.com/)
        map_tile_type = kwargs.get('map_tile_type', 'terrain-background')
        tiler = cimgt.Stamen(map_tile_type)
        mercator = tiler.crs

        # Figure Size
        fig = plt.figure(figsize=(15, 20), constrained_layout=False)
        gs = gridspec.GridSpec(40, 30, figure=fig)
        ax = fig.add_subplot(gs[:30, :], projection=mercator)
        ax.set_extent([sites.longitude.max()+bound_, sites.longitude.min()-bound_, sites.latitude.min()-bound_, sites.latitude.max()+bound_], crs=ccrs.PlateCarree())
        
        scale_arrow = kwargs.get('scale_arrow', 40)
        length = kwargs.get('length', 25)
        scale_loc = kwargs.get('scale_loc', (0.5, 0.05))
        hei_ = kwargs.get('hei_', 5)
        map_tile_type = kwargs.get('map_tile_type', 'terrain-background')

        strain_viz.ellipse_plot(self, V=V, arrows=arrows, color_list=color_list, tiler_size=tiler_size, map_tile_type=map_tile_type,
                                hei_=hei_, scale_arrow=scale_arrow, length=length, scale_loc=scale_loc, loc_=loc_, ax=ax, fig=fig)

        fig.canvas.draw()

        ax1 = fig.add_subplot(gs[30:34, 23:])

        image = kwargs.get('image', "https://www.unavco.org/education/resources/lib/images/unavco-logo-red-white-shadow.png")

        strain_viz.unavco_logo(image=image, ax=ax1)

        ax1_1 = fig.add_subplot(gs[31:33, :23])

        title_ = kwargs.get('title', "GPS Triangle-Strain Map Using UNAVCO PBO Data")
        fontsize_ = kwargs.get('fontsize', 24)
        ha_ = kwargs.get('ha', 'left')
        va_ = kwargs.get('va', 'center')
        
        strain_viz.map_title(title=str(title_), fontsize=fontsize_, ha=ha_, va=va_, ax=ax1_1)

        ax2 = fig.add_subplot(gs[30:40, 10:24])

        strain_viz.quiver_legend(self, sites=sites, size=size, label=label, scale_arrow=scale_arrow, ax=ax2)

        ax3 = fig.add_subplot(gs[33:37, 1:10])

        strain_viz.strain_legend(ax=ax3)

        ax4 = fig.add_subplot(gs[38:, :])

        strain_viz.table_data(self, sites=sites, ax=ax4)

        ax5 = fig.add_subplot(gs[34:36, 21:])

        strain_viz.speed_data(self, sites=sites, ax=ax5)
        
        save_fig = kwargs.get('save_fig', None)
        
        if save_fig is not None:
            plt.savefig(str(save_fig), edgecolor='k', bbox_inches='tight')
    
    def unavco_logo(**kwargs):
        im_read = kwargs.get('image', "https://www.unavco.org/education/resources/lib/images/unavco-logo-red-white-shadow.png")
        a = plt.imread(im_read)
        plt.imshow(a, aspect='equal')
        axis('off')
        
    def map_title(**kwargs):
        ax = kwargs.get('ax', None)

        if ax is None:
            fig = plt.figure(figsize=(5, 1.5))
            ax = fig.add_subplot(1, 1, 1)

        title_ = kwargs.get('title', "GPS Triangle-Strain Map Using UNAVCO PBO Data")
        fontsize_ = kwargs.get('fontsize', 20)
        ha_ = kwargs.get('ha', 'center')
        va_ = kwargs.get('va', 'top')
        xy_ = kwargs.get('xy', (0.0, 0.5))
        ax.annotate(str(title_), xy=xy_, va=va_, ha=ha_, fontsize=fontsize_)
        ax.axis('off')
    
    def ellipse_subplot(self, V, **kwargs):
        ax = kwargs.get('ax', None)

        if ax is None:
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(1, 1, 1)

        strain_viz.def_ellipse(self, V)
        ax.set_title("Infinitesimal Strain Ellipse", x=0.5, y=1.05, fontsize=16, fontweight='light')

        sites_h = []
        colors = ['tomato', 'cornflowerblue']
        strain_ = ['$S_{1H}$', '$S_{2H}$']
        for i in range(2):
            site_0 = Line2D([0], [0], color=colors[i], linestyle='-', linewidth=1.5, fillstyle='full', label=strain_[i])
            sites_h.append(site_0)

        leg = ax.legend(handles=[sites_h[0], sites_h[1]], ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.1), fontsize="x-large", frameon=False)
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(0.5)
        leg.get_frame().set_alpha(0.5)
        ax.axis('off')
        
    def contraction(**kwargs):
        ax = kwargs.get('ax', None)

        if ax is None:
            fig = plt.figure(figsize=(5, 2.5))
            ax = fig.add_subplot(1, 1, 1)

        ax.spines['left'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('center')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        max_strain = kwargs.get('max_strain', 300)
        min_strain = kwargs.get('min_strain', 0.1)

        rot0 = mtrans.Affine2D().rotate_deg(0)

        x0, y0 = rot0.transform_point((0.35, -strain_viz.scale_arrow(max_strain, [min_strain, max_strain], [0.25, 0.75]) + 0.75))
        x1, y1 = rot0.transform_point((-0.35, -strain_viz.scale_arrow(min_strain, [min_strain, max_strain], [0.25, 0.75]) + 0.5))

        sz_e1 = strain_viz.scale_arrow(max_strain, [min_strain, max_strain], [40, 80])
        sz_e2 = strain_viz.scale_arrow(min_strain, [min_strain, max_strain], [40, 80])

        x = np.array([-0.35, 0.35])
        y_1 = np.array([0.48, 0.73])
        y_2 = np.array([y1+0.01, y0+0.01])

        plt.plot((-0.35, 0.35), (0.48, 0.73), color='slategrey', linewidth=1, linestyle='--', marker=',')
        plt.plot((-0.35, 0.35), (y1+0.01, y0+0.01), color='slategrey', linewidth=1, linestyle='--', marker=',')
        plt.fill_between(x, y_1, y_2, where=(y_1 > y_2), color='slategrey', alpha=0.15, interpolate=True)

        ax.annotate("",
                    xy=(0.35, 0.75),
                    xytext=(x0, y0), textcoords='data',
                    size=sz_e1, va="center", ha="center", color='k',
                    arrowprops=dict(arrowstyle="simple, head_length=0.35,head_width=0.5,tail_width=0.2", fc="k", ec='k', lw=2)
                    )

        ax.annotate("",
                    xy=(-0.35,0.5),
                    xytext=(x1, y1),
                    size=sz_e2, va="center", ha="center", color='k',
                    arrowprops=dict(arrowstyle="simple, head_length=0.35,head_width=0.5,tail_width=0.2", fc="k", ec='k', lw=2)
                    )

        ax.annotate("Infinitesimal Strain (Contraction)", xy=(0.0, 0.9), xycoords="data",
                          va="top", ha="center", fontsize=16)

        ax.annotate("%s\nnano-strain" % (min_strain), xy=(-0.75, 0.3), xycoords="data",
                          va="center", ha="center", fontsize=12)

        ax.annotate("%s\nnano-strain" % (max_strain), xy=(0.75, 0.3), xycoords="data",
                          va="center", ha="center", fontsize=12)


        ax.set_xlim([-1,1])
        ax.set_ylim([0,1])
        ax.axis('off')
        
    def elongation(**kwargs):
        ax = kwargs.get('ax', None)

        if ax is None:
            fig = plt.figure(figsize=(5, 2.5))
            ax = fig.add_subplot(1, 1, 1)

        ax.spines['left'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('center')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        max_strain = kwargs.get('max_strain', 300)
        min_strain = kwargs.get('min_strain', 0.1)

        scale_arrow_percent_0 = strain_viz.scale_arrow(max_strain, [min_strain, max_strain], [0.2, 0.6])
        boxstyle0_d = f"darrow,pad=%s" % (scale_arrow_percent_0)

        scale_arrow_percent_1 = strain_viz.scale_arrow(min_strain, [min_strain, max_strain], [0.2, 0.6])
        boxstyle1_d = f"darrow,pad=%s" % (scale_arrow_percent_1)

        sz_e1_d = strain_viz.scale_arrow(max_strain, [min_strain, max_strain], [10, 15])
        sz_e2_d = strain_viz.scale_arrow(min_strain, [min_strain, max_strain], [10, 15])

        x = np.array([0.85, 0.35, -0.21, -0.32])
        y_2 = np.array([scale_arrow_percent_1+0.1, scale_arrow_percent_0 + 0.0975, 0.65, 0.125])

        ax.fill(x, y_2, color='slategrey', alpha=0.15)

        plt.plot((-0.21, -0.32), (0.65, 0.125), color='slategrey', linewidth=1, linestyle='--', marker=',')
        plt.plot((0.8, 0.35), (scale_arrow_percent_1+0.15, scale_arrow_percent_0 + 0.0975), color='slategrey', linewidth=1, linestyle='--', marker=',')

        bbox_props2 = dict(boxstyle=boxstyle1_d, fc="w", ec="k", lw=3)

        sz_text1 = "---------------" + ('-' * int(20*float(scale_arrow_percent_1)))

        ax.text(0.1, 0.68, sz_text1, ha="center", va="top", rotation=0,
                size=sz_e2_d, color='w', bbox=bbox_props2)

        sz_text0 = "---------------" + ('-' * int(20*float(scale_arrow_percent_0)))

        bbox_props3 = dict(boxstyle=boxstyle0_d, fc="w", ec="k", lw=3)
        ax.text(0.35, 0.2, sz_text0, ha="center", va="top", rotation=0,
                size=sz_e1_d, color='w', bbox=bbox_props3)

        ax.annotate("Infinitesimal Strain (Elongation)", xy=(0.0, 0.925), xycoords="data",
                          va="top", ha="center", fontsize=16, fontweight='book')

        ax.annotate("%s\nnano-strain" % (min_strain), xy=(-0.65, 0.63), xycoords="data",
                      va="center", ha="center", fontsize=12)

        ax.annotate("%s\nnano-strain" % (max_strain), xy=(-0.65, 0.05), xycoords="data",
                      va="center", ha="center", fontsize=12)

        bboxprops = dict(boxstyle="round,pad=1", facecolor='white', edgecolor='black', lw=3)
        ax.annotate("", xy=(-0.65, 0.05), xycoords="data",
                      va="center", ha="center", fontsize=12, bbox=bboxprops)

        ax.set_xlim([-1,1])
        ax.set_ylim([0, 1])
        ax.axis('off')
        
    def table_data(self, sites, **kwargs):
        ax = kwargs.get('ax', None)
        fontsize = kwargs.get('fontsize', 11.25)
        scale = kwargs.get('fontsize', 1.75)

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

        table = ax.table(cellText=sites.round(6).values, colLabels=sites.columns, cellLoc='center', rowLoc='center',loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)
        table.scale(1, scale)
        ax.axis('off')
        
    def speed_data(self, sites, **kwargs):
        ax = kwargs.get('ax', None)

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

        sites_t = sites.copy().drop(['E uncertainty (mm/yr)', 'N uncertainty (mm/yr)'], axis=1)
        sites_t.columns = ['sites', 'longitude', 'latitude', 'east_v', 'north_v']
        sites_t['Speed (mm/yr)'] = sites_t[['east_v', 'north_v']].apply(lambda x: np.sqrt((x.east_v**2)+(x.north_v**2)), axis=1)
        sites_t = sites_t[['sites', 'Speed (mm/yr)']]

        table = ax.table(cellText=sites_t.round(6).values, colLabels=sites_t.columns, cellLoc='center', rowLoc='center',loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11.25)
        table.scale(1, 1.75)
        ax.axis('off')
        
    def quiver_legend(self, sites, **kwargs):
        ax = kwargs.get('ax', None)
        size = kwargs.get('size', 10)
        label = kwargs.get('label', '10 mm/yr')
        scale_arrow = kwargs.get('scale_arrow', 40)

        if ax is None:
            #fig_ = plt.figure(figsize=(5, 5))
            fig_ = plt.figure()
            ax = fig_.add_subplot(1, 1, 1)

        Q = ax.quiver(sites['E velocity (mm/yr)'], sites['N velocity (mm/yr)'], scale=scale_arrow, width=0.0175, headwidth=3.5, color='k')
    
        ax.clear()

        p_fancy = FancyBboxPatch((0.115, 0.415),
                                 0.59, 0.17,
                                 boxstyle="square,pad=0.05", 
                                 fc='w', ec='k', lw=1, alpha=0.25)
        ax.add_patch(p_fancy)

        annotate("Velocity Relative to SNARF", xy=(0.4, 0.6), xycoords="data",
                 va="top", ha="center", fontsize=14, fontweight='book')
        ax.quiverkey(Q, 0.45, 0.45, size, label, labelpos='E', fontproperties=dict(size=12.5), labelsep=0.2,
                      coordinates='axes')
        ax.axis('off')
        
    def strain_legend(**kwargs):
        ax = kwargs.get('ax', None)

        if ax is None:
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(1, 1, 1)

        sites_h = []
        colors = ['tomato', 'cornflowerblue']
        strain_ = ['$S_{1H}$', '$S_{2H}$']
        for i in range(2):
            site_0 = Line2D([0], [0], color=colors[i], linestyle='-', linewidth=1.5, fillstyle='full', label=strain_[i])
            sites_h.append(site_0)

        site_1 = Line2D([0], [0], marker='$\u25CC$', color='w', linestyle='--', markeredgecolor='slategrey',
                            markeredgewidth=0.5, label='Initial State', markerfacecolor='slategrey', markersize=20)

        site_2 = Line2D([0], [0], marker='o', color='w', linestyle='--', markeredgecolor='k',
                            markeredgewidth=1.1, label='Strain Ellipse', markerfacecolor='w', markersize=18)

        leg = ax.legend(handles=[sites_h[0], site_1, sites_h[1], site_2], ncol=2, loc='center', fontsize="x-large", frameon=True, title="Strain Ellipse Legend")
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(0.5)
        leg.get_frame().set_alpha(0.5)
        plt.setp(leg.get_title(),fontsize=14)
        ax.axis('off')
