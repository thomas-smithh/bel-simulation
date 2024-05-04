import numpy as np
import pandas as pd
import joblib
import contextlib
from tqdm import tqdm
import numba as nb
import scipy.spatial as sp
from scipy.integrate import solve_ivp
import random

@contextlib.contextmanager
def tqdm_joblib(
    tqdm_object
):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

@nb.njit(nb.float64[:](nb.float64[:, :], nb.int32, nb.int32, nb.int32, nb.float64[:]))
def partials(
    positions: np.array, 
    i: int, 
    j: int, 
    k: int, 
    area: np.array
) -> np.array:
    """
    Calculate the partial derivatives required when calculating the force due to preferred areas (BM contribution).
    Args:
        positions (np.array): A 2D numpy array of shape (N_cells, 3) representing the positions of cell centers.
            positions[i,:] = x,y,z positions of center of cell i.
        i (int): The index of the first vertex of the simplex.
        j (int): The index of the second vertex of the simplex.
        k (int): The index of the third vertex of the simplex.
        area (np.array): A numpy array containing the area of the simplex defined by vertices i, j, and k.
    Returns:
        np.array: A numpy array representing 1/(2*area) * partial derivative of the area with respect to xi, yi, and zi.
    """
    x_i = positions[i,0] 
    y_i = positions[i,1]
    z_i = positions[i,2]
    x_j = positions[j,0] 
    y_j = positions[j,1]
    z_j = positions[j,2]
    x_k = positions[k,0] 
    y_k = positions[k,1]
    z_k = positions[k,2] 

    partial_xi =(y_j - y_k)*((x_j-x_i)*(y_k-y_i) - (x_k-x_i)*(y_j-y_i) )+ (z_k-z_j)*(-(x_j-x_i )*(z_k-z_i) + (x_k-x_i)*(z_j-z_i))
    partial_yi =(-x_j + x_k)*((x_j-x_i)*(y_k-y_i) - (x_k-x_i)*(y_j-y_i))+ (z_j - z_k)*((y_j-y_i)*(z_k-z_i) - (y_k-y_i)*(z_j-z_i))
    partial_zi =(-x_j + x_k)*((x_j-x_i)*(z_k-z_i) - (x_k-x_i)*(z_j-z_i)) + (y_k-y_j)*((y_j-y_i)*(z_k-z_i) - (y_k-y_i)*(z_j-z_i))
    
    return 1/(2*area)* np.array([partial_xi, partial_yi, partial_zi])

@nb.njit(nb.float64[:, :](nb.int32[:, :], nb.int32))
def get_neighbours(
    simplices: np.array, 
    N_bodies: int
) -> np.array:
    """
    Determine the neighbors for each body using simplices calculated from a Delaunay triangulation.
    This function creates a matrix where each row corresponds to a body and each column contains the indices of its neighbors.
    The matrix is filled with NaNs initially, and valid neighbor indices are populated based on the simplices provided.

    Args:
        simplices (np.array): A 2D numpy array of simplices of shape (N_simplices, 4), calculated as simplicies = sp.Delaunay(positions).simplices.
            Indices of the points forming the simplices in the triangulation.
        N_bodies (int): The total number of bodies in the system.
            Before lumen formation, N_bodies = N_cells. After lumen formation, N_bodies = N_cells + 1 (lumen)

    Returns:
        neighbours (np.array): A 2D numpy array where each row represents a body and each entry in the row is an index of a neighbor.
            The array dimensions are (N_bodies, N_bodies).
            neighbours[i,:] = indices of cells neighbouring cell i
            Neighbours of each cell are filled left to right, so neighbours[;,j] = np.NaN indicates no further neighbours
    """
    neighbours = np.empty((N_bodies, N_bodies))
    neighbours[:] = np.nan
    for i in range(np.shape(simplices)[0]):
        simplex = simplices[i]
        for j in range(4):
            for k in range(4):
                if j != k and simplex[j] not in neighbours[simplex[k]]:
                    for m in range(N_bodies):
                        if np.isnan(neighbours[simplex[k], m]):
                            neighbours[simplex[k], m] = simplex[j]
                            break
    return neighbours

@nb.njit(nb.types.UniTuple(nb.float64[:, :], 2)(nb.float64[:, :], nb.float64[:, :], nb.float64[:, :], nb.float64[:, :], nb.float64, nb.float64[:, :], nb.int32, nb.int32[:, :], nb.float64[:, :], nb.float64[:, :], nb.float64[:, :], nb.float64[:, :], nb.int32, nb.float64, nb.float64))
def calculate_force(
    positions: np.array, 
    neighbours: np.array, 
    ages: np.array, 
    lifetimes: np.array, 
    alpha: float, 
    morse_force_matrix: np.array, 
    N_bodies: int, 
    simplices: np.array, 
    areas: np.array, 
    equations: np.array, 
    bm_pressure_force_matrix: np.array, 
    bm_area_force_matrix: np.array, 
    beta: float, 
    P_star: float, 
    A_eq_star: float
) -> np.array:
    """
    """
    for ii in range(N_bodies):
        for jj in neighbours[ii]:
            if np.isnan(jj) or ii <= jj:
                continue
            else:   
                jj = int(jj)
                r_ij_star = positions[jj,:] - positions[ii,:]
                r_mag = np.sqrt(np.dot(r_ij_star,r_ij_star))
                J = 2 + ((np.cbrt(2)-1)* ((ages[ii]/lifetimes[ii])+(ages[jj]/lifetimes[jj]))) ###################POSSIBLY SPEED THIS UP BY CALCULATING ALL JS AT ONCE IN MATRIX
                F_morse_mag = -2*(np.exp(alpha*(r_ij_star - J)) - np.exp(2*alpha*(r_ij_star-J)))
                F_morse = F_morse_mag *  (r_ij_star/r_mag)
                morse_force_matrix[ii,:] -= F_morse
                morse_force_matrix[jj,:] += F_morse

    # CALCULATE FORCE DUE TO PREFERRED SIMPLEX AREAS (BM CONTRIBUTION)
    for count, face in enumerate(simplices): # face has the form [i j k], where i j and k are the indices of the cells that define one simplex
        areas[count] = 0.5* np.linalg.norm(np.cross(positions[face[1]]-positions[face[0]],positions[face[2]]-positions[face[0]])) #area of each face
        for index in range(3):
            bm_pressure_force_matrix[face[(index%3)]] += (beta/3)* P_star * areas[count] * equations[count,0:3]
            bm_area_force_matrix[face[index]] -= beta * (areas[count]-A_eq_star) * partials(positions, face[index], face[(index+1)%3], face[(index+2)%3], areas[count])     

    force_matrix = bm_pressure_force_matrix + bm_area_force_matrix + morse_force_matrix

    return force_matrix, areas

class Simulation:

    def __init__(
        self,
        N_bodies=4,
        boxsize=5,
        r_min=1,
        mean_lifetime=5,
        delta_t_max=0.001
    ):
        """
        """

        self.all_forces = []
        self.average_areas = []
        self.all_volumes = []
        self.all_positions = []
        self.all_ages = []
        self.all_preferred_areas = []

        self.current_areas = np.inf

        self.all_t_values = [0, 0]
        self.last_event_time = 0
        self.fail_cause = 0
        self.lumen_status = False
        

        self.N_bodies = N_bodies
        self.boxsize = boxsize
        self.r_min = r_min
        self.mean_lifetime = mean_lifetime
        self.t_max = self.mean_lifetime * 4
        self.delta_t_max = delta_t_max

        self.lifetime_std = self.mean_lifetime/100
        self.lifetimes = np.random.normal(self.mean_lifetime, self.lifetime_std, size=(self.N_bodies, 1))
        self.ages = self.lifetimes * np.random.rand(self.N_bodies, 1)

        self.positions = np.concatenate([self.fibonacci_sphere(self.N_bodies-1, 2), np.array([0, 0, 0]).reshape(1,3)])
        self.force_matrix = np.zeros((self.N_bodies, 3))
        self.hull = sp.ConvexHull(self.positions)
        self.radii = np.zeros_like(self.ages)
        self.volumes = np.zeros_like(self.ages)

        self.calculate_radius_from_age()
        self.calculate_volume_from_radius()
        self.lumen_volume = self.volumes[-1]
        self.lumen_radius = self.radii[-1]
        self.all_lumen_volumes = [self.lumen_volume]

        self.events.terminal = True

    def calculate_radius_from_age(
        self
    ):
        self.radii = self.r_min * (1 + ((np.cbrt(2)-1)  * (self.ages/self.lifetimes)))
        self.lumen_radius = self.radii[-1]

    
    def calculate_volume_from_radius(
        self
    ):
        self.volumes = 4/3 * np.pi * self.radii**3
        self.lumen_volume = self.volumes[-1]

    @staticmethod
    def fibonacci_sphere(
        samples=1, 
        radius=2
    ):
        """
        """

        points = []
        phi = np.pi * (3. - np.sqrt(5.)) 

        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2 
            radius_c = np.sqrt(1 - y * y)

            theta = phi * i 

            x = np.cos(theta) * radius_c
            z = np.sin(theta) * radius_c

            points.append((radius * x, radius * y, radius * z))

        return np.array(points)

    def calculate_total_forces(
        self
    ):
        """
        """
        
        self.hull = sp.ConvexHull(self.positions)
        dela = sp.Delaunay(self.positions)
        self.neighbours = np.empty((self.N_bodies, self.N_bodies))
        self.neighbours[:] = np.nan
        self.neighbours = get_neighbours(dela.simplices, self.N_bodies)

        self.force_matrix = np.zeros((self.N_bodies, 3)) 
        self.bm_pressure_force_matrix = np.zeros((self.N_bodies, 3))
        self.bm_area_force_matrix = np.zeros((self.N_bodies,3))
        self.morse_force_matrix = np.zeros((self.N_bodies, 3))
        
        self.areas = np.zeros([len(self.hull.simplices),1])
        self.force_matrix, self.areas = calculate_force(
            self.positions, 
            self.neighbours, 
            self.ages, 
            self.lifetimes, 
            self.alpha, 
            self.morse_force_matrix, 
            self.N_bodies, 
            self.hull.simplices, 
            self.areas, 
            self.hull.equations, 
            self.bm_pressure_force_matrix, 
            self.bm_area_force_matrix, 
            self.beta, 
            self.P_star, 
            self.A_eq_star
        )

        self.all_positions.append(self.positions)
        self.all_forces.append(self.force_matrix)
        self.all_volumes.append(self.hull.volume)
        self.current_areas = self.areas
        self.average_areas.append(self.areas.mean())
        self.all_lumen_volumes.append(self.lumen_volume)
        self.all_preferred_areas.append(self.A_eq_star)

    @staticmethod
    def r_dot(
        t,
        r,
        self
    ): 
        """
        """

        self.positions = np.asarray(r).reshape(self.N_bodies,3)
        time_increment = self.all_t_values[-1]-self.all_t_values[-2]

        self.calculate_total_forces()
        current_ages = self.ages   
        self.all_t_values.append(t)
        self.A_eq_star = (self.hull.area / (self.hull.simplices.shape[0])) * self.A_eq_star_scaling
        self.all_preferred_areas.append(self.A_eq_star)

        if self.lumen_status == False:
            self.ages = current_ages + time_increment
        else:
            self.ages[:-1] = current_ages[:-1] + time_increment
            self.ages[-1] = current_ages[-1]  
            
        self.all_ages.append(self.ages)
        round_percent = int(round((self.all_t_values[-1]/self.t_max)*50))
        drdt = self.force_matrix.flatten().tolist()
        return drdt

    @staticmethod
    def events( 
        t, 
        r,
        self
    ):
        """
        """

        if (any(self.current_areas) <  self.A_eq_star) or (self.positions.shape[0]<=3):
            return 0

        elif self.last_event_time != 0 and (self.all_t_values[-1] - self.last_event_time) <= self.mean_lifetime/1000:
            return 1

        elif any(self.ages[0:-1] > self.lifetimes[0:-1]): 
            return 0

        else:
            distances = np.linalg.norm(self.positions - self.positions[-1], axis=1) 
            cells_inside = [i for i in range(self.N_bodies-1) if distances[i] <= self.radius_scaling * self.lumen_radius]
            if len(cells_inside) > 1:
                return 0
            else:
                return 1

    def cell_joins_lumen(
        self
    ):

        new_lumen_volume = 0
        numerator, denominator = 0, 0

        self.calculate_radius_from_age()
        distances = np.linalg.norm(self.positions - self.positions[-1], axis=1) 

        bodies_in_centre = [i for i in range(self.N_bodies) if distances[i] <= (self.radius_scaling * self.lumen_radius)] 

        new_exterior_cells = [i for i in range(self.N_bodies) if i not in bodies_in_centre]
        new_cell_number = len(new_exterior_cells)
        
        if new_cell_number >= 2:
            if self.lumen_status == False:
                N_cells = len(self.hull.vertices) 
                new_positions = np.zeros((N_cells + 1, 3))
                new_ages = np.zeros((N_cells + 1, 1))
                new_lifetimes = np.zeros((N_cells + 1, 1))
                bodies_in_centre = [i for i in range(self.N_bodies) if i not in self.hull.vertices]

                for i, body in enumerate(bodies_in_centre):
                    mass = self.r_min * (4/3) * np.pi * (1+ ((np.cbrt(2)-1)*self.ages[body,0]/self.lifetimes[body,0]))**3
                    if i== 0:
                        new_lumen_volume += mass
                    else:
                        new_lumen_volume += self.volume_scaling * mass
            else:
                new_N_bodies = new_cell_number + 1 
                new_positions = np.zeros((new_N_bodies, 3))
                new_ages = np.zeros((new_N_bodies, 1))
                new_lifetimes = np.zeros((new_N_bodies, 1))

                for i, body in enumerate(bodies_in_centre):
                    if body != self.N_bodies-1:
                        mass = self.r_min * (4/3) * np.pi * (1+ ((np.cbrt(2)-1)*self.ages[body,0]/self.lifetimes[body,0]))**3
                        added_mass = mass * self.volume_scaling
                        new_lumen_volume += added_mass
                    else:
                        mass = self.r_min * (4/3) * np.pi * (1+ ((np.cbrt(2)-1)*self.ages[-1,0]/self.mean_lifetime))**3
                        added_mass = mass
                        new_lumen_volume += added_mass 
    
            for i, body in enumerate(new_exterior_cells):  
                mass = self.r_min * (4/3) * np.pi * (1+ ((np.cbrt(2)-1)*self.ages[body,0]/self.lifetimes[body]))**3
                numerator += mass * self.positions[body]
                denominator += mass

            for count, cell_index in enumerate(new_exterior_cells):
                new_positions[count] = self.positions[cell_index]
                new_ages[count] = self.ages[cell_index]
                new_lifetimes[count] = self.lifetimes[cell_index]

            lumen_scale_age = (self.mean_lifetime/(np.cbrt(2) -1)) *(((1/self.r_min) * np.cbrt(3*new_lumen_volume * 0.25 * (1/np.pi)))-1)
            self.lumen_radius = (0.75 * new_lumen_volume * (1/np.pi))**(1/3)
            self.lumen_volume = new_lumen_volume
            
            new_ages[-1] = lumen_scale_age 
            new_lifetimes[-1] = self.mean_lifetime
            new_positions[-1,:] = numerator/denominator 

            self.ages = new_ages
            self.lifetimes = new_lifetimes
            self.N_bodies = self.ages.shape[0]
            self.positions = new_positions
            self.lumen_status = True
        
            try:
                self.hull=sp.ConvexHull(self.positions)
                self.last_event_time = self.all_t_values[-1]
                self.all_lumen_volumes.append(self.lumen_volume)
            except:
                self.fail_cause = "all_lumen"

        else:
            self.fail_cause = "all_lumen"

    def perform_divisions(
        self
    ): 
        """
        """

        N_old = self.N_bodies
        old_ages = self.ages
        old_positions = self.positions

        if self.lumen_status == False:
            number_dividing = self.ages[self.ages > self.lifetimes].size
            enumerator = self.ages
            start_index = 0
        else:
            number_dividing = self.ages[:-1][self.ages[:-1] > self.lifetimes[:-1]].size
            enumerator = self.ages[:-1]
            start_index = -1
        
        self.N_bodies =  N_old + number_dividing
        new_positions = np.zeros((self.N_bodies,3))
        new_ages = np.zeros((self.N_bodies, 1))
        new_lifetimes = np.zeros((self.N_bodies, 1))
        
        j = 0 
        for i, age in enumerate(enumerator):
            if age < self.lifetimes[i]:
                new_positions[i] = self.positions[i]
                new_ages[i] = self.ages[i]
                new_lifetimes[i] = self.lifetimes[i]
            else:
                new_ages[i] = 0
                new_lifetimes[i] = np.random.normal(self.mean_lifetime, self.lifetime_std)
                new_ages[N_old + j+ start_index] = 0
                new_lifetimes[N_old + j + start_index] = np.random.normal(self.mean_lifetime, self.lifetime_std)

                theta_rand = np.random.uniform(0,360)
                phi_rand = np.random.uniform(0,360)

                new_positions[i][0] = self.positions[i][0] + self.r_min*np.cos(phi_rand)* np.sin(theta_rand)
                new_positions[i][1] = self.positions[i][1] +  self.r_min * np.sin(phi_rand) * np.sin(theta_rand)
                new_positions[i][2] = self.positions[i][2] +  self.r_min * np.cos(theta_rand)
                new_positions[N_old + j + start_index][0] =  new_positions[i][0]-(2*self.r_min*np.cos(phi_rand)* np.sin(theta_rand))
                new_positions[N_old + j + start_index][1] = new_positions[i][1]-(2*self.r_min * np.sin(phi_rand) * np.sin(theta_rand))
                new_positions[N_old + j+ start_index][2] = new_positions[i][2] - (2*self.r_min * np.cos(theta_rand))        
                j += 1

        if self.lumen_status == True:
            new_ages[-1] = old_ages[-1]
            new_lifetimes[-1] = self.mean_lifetime
            new_positions[-1] = old_positions[-1]

        self.positions = new_positions
        self.ages = new_ages
        self.lifetimes = new_lifetimes
        self.hull = sp.ConvexHull(self.positions)
        self.N_bodies = self.ages.shape[0]
        self.last_event_time = self.all_t_values[-1]

    def execute(
        self,
        beta,
        alpha,
        P_star,
        volume_scaling,
        radius_scaling,
        A_eq_star_scaling,
        A_eq_star=0.1,
        write_results=False,
        write_path="C:\\Users\\Tom\\Documents\\Bel PhD\\Bel_Simulation\\outputs",
        run_number=0,
        alter='all'
    ):
        """
        """

        self.beta = beta
        self.alpha = alpha
        self.A_eq_star_scaling = A_eq_star_scaling
        self.A_eq_star = A_eq_star
        self.P_star = P_star
        self.volume_scaling = volume_scaling
        self.radius_scaling = radius_scaling

        t_min = 0
        while t_min < self.t_max:
            sol = solve_ivp(
                fun=self.r_dot, 
                t_span=(t_min, self.t_max), 
                y0=self.positions.flatten(), 
                method='RK45', 
                first_step=0.0003, 
                max_step=self.delta_t_max, 
                events=(self.events), 
                args=(self,)
            )

            t_min = max(sol.t)
            
            if t_min > self.t_max:
                self.fail_cause = "success"
                break

            if any (self.current_areas) <  self.A_eq_star:
                self.fail_cause = "small_area"
                break

            if (self.lumen_status == True) & (self.positions.shape[0]<=3):
                self.fail_cause = "all_lumen"
                break

            if self.lumen_status == False:
                if len(self.hull.vertices) != self.N_bodies:
                    self.cell_joins_lumen()
                    if self.fail_cause == "all_lumen":
                        break
                elif any(self.ages >= self.lifetimes):   
                    self.perform_divisions()

            else:
                if len([i for i in range(self.N_bodies) if np.linalg.norm(self.positions - self.positions[-1], axis=1) [i] <= (self.lumen_radius*self.radius_scaling)]) > 1:
                    self.cell_joins_lumen()
                    if self.fail_cause == "all_lumen":
                        break
                elif any(self.ages[0:-1] > self.lifetimes[0:-1]):
                    self.perform_divisions()
                else:
                    self.fail_cause = "unknown"

        del self.all_t_values[0:2]

        self.results = pd.DataFrame()
        self.results = pd.DataFrame(list(zip(*[self.all_t_values, self.all_volumes, self.average_areas, self.all_preferred_areas])))
        self.results.columns = ["t", "Cluster_Vol", "Areas", "Preferred_area"]
        self.results["Run_No"] = run_number
        self.results["r_min"] = self.r_min
        self.results["beta"] = self.beta
        self.results["alpha"] = self.alpha
        self.results["A_eq_star_scaling"] = self.A_eq_star_scaling
        self.results["P_star"] = self.P_star
        self.results["mean_lifetime"] = self.mean_lifetime
        self.results["lumen_volume_scaling"] = self.volume_scaling
        self.results["lumen_radius_scaling"] = self.radius_scaling
        self.results["end_reason"] = self.fail_cause
        
        if write_results:
            self.results.to_parquet("{}\\Alter{}_Run{}.parquet".format(write_path, alter, run_number))
            np.save('{}\\Alter{}_positions_{}.npy'.format(write_path, alter, run_number), np.array(self.all_positions, dtype=object), allow_pickle=True)
            np.save('{}\\Alter{}_ages_{}.npy'.format(write_path, alter, run_number), np.array(self.all_ages, dtype=object), allow_pickle=True)