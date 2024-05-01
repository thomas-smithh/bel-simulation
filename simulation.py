import numpy as np
import pandas as pd
import joblib
import contextlib
from tqdm import tqdm

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
        neighbors=None,
        force_matrix=None,
        bm_area_force_matrix=None,
        bm_pressure_force_matrix=None,
        morse_force_matrix=None,
        hull=None,
        N_bodies=None,
        all_N_bodies=None,
        A_eq_star=None,
        inner_pressure=None,
        positions=None,
        ages=None,
        A_eq_star=None,
        A_eq_star_scaling=None,
        lumen_radius=None,
        lifetimes=None,
        lumen_volumes=None,
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
        self.all_lumen_volumes = []

        self.neighbors = neighbors
        self.force_matrix = force_matrix
        self.hull = hull
        
        self.bm_area_force_matrix = bm_area_force_matrix
        self.bm_pressure_force_matrix = bm_pressure_force_matrix
        self.morse_force_matrix = morse_force_matrix
        self.N_bodies = N_bodies
        self.all_N_bodies = all_N_bodies
        self.A_eq_star = A_eq_star
        self.inner_pressure = inner_pressure
        self.positions = positions
        self.ages = ages
        
        self.A_eq_star = A_eq_star
        self.A_eq_star_scaling = A_eq_star_scaling
        self.last_event_time = last_event_time
        self.lumen_radius = lumen_radius
        self.lifetimes = lifetimes
        self.lumen_volumes = lumen_volumes

        self.events.terminal = True

    def calculate_total_forces(
        self
    ):
        """
        """
        global neighbours, force_matrix, hull
        global current_areas
        global all_forces, average_areas, all_volumes, all_lumen_volumes, all_preferred_areas, all_positions
        global bm_area_force_matrix, bm_pressure_force_matrix, morse_force_matrix, all_N_bodies, A_eq_star, inner_pressure

        hull = sp.ConvexHull(positions)
    ###############################################################################################################
    ## CALCULATE THE NEIGHBOURS
        dela = sp.Delaunay(positions)
        self.neighbours = np.empty((N_bodies, N_bodies))
        self.neighbours[:] = np.nan
        self.neighbours = get_neighbours(dela.simplices, N_bodies)

        ###############################################################################################################

        ###############################################################################################################
        # CALCULATE THE FORCES
        # CREATE A BLANK FORCE MATRIX
        self.force_matrix = np.zeros((N_bodies, 3)) #N_bodies = N_cells + N_lumen(=0/1)
        self.bm_pressure_force_matrix = np.zeros((N_bodies, 3))
        bm_area_force_matrix = np.zeros((N_bodies,3))
        morse_force_matrix = np.zeros((N_bodies, 3))
        
        # CALCULTE MORSE FORCE + FORCE DUE TO INNER PRESSURE
        
        self.areas = np.zeros([len(hull.simplices),1])

        self.force_matrix, self.areas = calculate_force(positions, neighbours, ages, lifetimes, alpha, morse_force_matrix, N_bodies, hull.simplices, areas, hull.equations, bm_pressure_force_matrix, bm_area_force_matrix, beta, P_star, A_eq_star)

        # TRACK PROPERTIES FOR PLOTTING (OPTIONAL)
        all_positions.append(positions)
        all_forces.append(force_matrix)
        all_volumes.append(hull.volume)
        current_areas = areas
        average_areas.append(areas.mean())
        all_lumen_volumes.append(lumen_volume)
        all_preferred_areas.append(A_eq_star)

    @staticmethod
    def r_dot(
        t,
        r,
        self
    ): 
        """
        """

        # r is a 1d list of the positions at a timepoint
        self.positions = np.asarray(r).reshape(N_bodies,3)
        time_increment = self.all_t_values[-1]-self.all_t_values[-2]

        self.calculate_total_forces()
        current_ages = self.ages   
        self.all_t_values.append(t)
        self.A_eq_star = (hull.area / (hull.simplices.shape[0])) * self.A_eq_star_scaling
        all_preferred_areas.append(A_eq_star)

        if self.lumen_status == False:
            self.ages = current_ages + time_increment
        else:
            self.ages[:-1] = current_ages[:-1] + time_increment
            self.ages[-1] = current_ages[-1]  
            
        self.all_ages.append(self.ages)
        round_percent = int(round((self.all_t_values[-1]/t_max)*50))
        print(round_percent*"|"+ (50-round_percent)*"-" + "t={}".format(round(self.all_t_values[-1],2)), end = "\r")
        ##########################################################
        drdt = force_matrix.flatten().tolist()
        return drdt

    @staticmethod
    def events( 
        t, 
        r,
        self
    ):
        ## UNPYSICAL EVENTS HAVE OCCURRED
        ##################SIMPLICES BELOW THE CUTOFF VALUE##################
        if any(self.current_areas) <  self.A_eq_star:
            print("AREAS TOO SMALL")
            return 0
        #######LUMEN IS ALL
        if positions.shape[0]<=3:
            print("LUMEN IS ALL")
            return 0
        # If there hasn't been much time since the last event, keep going without performing an action
        # Unless the last event time was 0 (i.e. there has been no event)
        if last_event_time != 0 and (all_t_values[-1] - last_event_time) <= mean_lifetime/1000 :
            return 1
        ################## TOO OLD ##################
        # Check if the age of any cell exceeds the lifetime
        # The "age" of the lumen is just used to calculate its radius, so its "age" can exceed the lifetime
        if self.lumen_status == False:
            too_old = int(not any (self.ages>=lifetimes))
        else:
            too_old = int(not any (self.ages[0:-1] > lifetimes[0:-1]))
        if too_old == 0:
            return 0

        ################## JOIN LUMEN ##################
        # If there's no lumen, any cell that falls inside the convex hull becomes the lumen
        # If there's already a lumen, cells join when their centre likes within (radius scaling * lumen radius) of the lumen centre
        if self.lumen_status == False:
            inside_hull = int(self.N_bodies == len(self.hull.vertices))
        else:
            distances = np.linalg.norm(self.positions - self.positions[-1], axis=1) 
            cells_inside = [i for i in range(self.N_bodies) if distances[i] <= radius_scaling * self.lumen_radius]
            if len(cells_inside) > 1:
                inside_hull = 0
            else:
                inside_hull = 1

        return inside_hull

    def cell_joins_lumen(
        self
    ):
        print("CELL JOINS LUMEN TRIGGERED AT TIME {} WHEN THERE ARE {} BODIES TOTAL AND LUMEN STATUS IS {}".format(self.all_t_values[-1], self.N_bodies, self.lumen_status))
        ####the lumen will still have an age (to define its volume)
        new_lumen_volume = 0
        numerator, denominator = 0,0

        # After the extra cells have joined the lumen, we need to have at least 2 cells remaining in order to position the lumen
        # and define a ConvexHull
        if self.lumen_status == False:
            new_exterior_cells = self.hull.vertices
        if self.lumen_status == True:
            existing_lumen_radius = self.r_min*(1 + ((2**(1/3))-1)  * ((self.ages[-1,0])/mean_lifetime))
            print("existing lumen radius = {}".format(existing_lumen_radius))
            print("radius scaling = {}".format(radius_scaling))
            # Calculate the distance between all bodies and the lumen (position[-1])
            distances = np.linalg.norm(self.positions - self.positions[-1], axis=1) 
            print("distances are {}".format(distances))
            bodies_in_centre = [i for i in range(self.N_bodies-1) if distances[i] <= (radius_scaling * existing_lumen_radius)] # this includes the current lumen
            bodies_in_centre.append(self.N_bodies-1) #so bodies in centre = existing lumen + inner cells
            new_exterior_cells = [i for i in range(self.N_bodies) if i not in bodies_in_centre]
        new_cell_number = len(new_exterior_cells)
        
        if new_cell_number >=2:
            # IF THIS IS THE CREATION OF THE LUMEN
            if self.lumen_status == False:
                print("CURRENTLY NO LUMEN")
                N_cells = len(self.hull.vertices) 
                new_positions = np.zeros((N_cells + 1, 3))
                new_ages = np.zeros((N_cells + 1, 1))
                new_lifetimes = np.zeros((N_cells + 1, 1))
                bodies_in_centre = [i for i in range(self.N_bodies) if i not in hull.vertices]
                print("THERE ARE {} BODIES IN THE CENTRE".format(len(bodies_in_centre)))
                print("CURRENT POSITIONS {}".format(self.positions))
                """
                Lumen creation is triggered by cells not in the convex hull, depending on the initial configuration, this could be >1 cell
                First cell contributes all of its volume to the lumen, and subsequent cells contribute (volume_scaling * cell_volume)
                """
                for i, body in enumerate(bodies_in_centre):
                    mass = self.r_min * (4/3) * np.pi * (1+ ((np.cbrt(2)-1)*self.ages[body,0]/self.lifetimes[body,0]))**3
                    if i== 0:
                        print("First cell in the new lumen is added, i = {}, mass = {}, positions ={}".format(i, mass, self.positions[body]))
                        new_lumen_volume += mass
                    else:
                        print("Extra cell added to new lumen, i = {}, mass = {}, positions ={}, volume scale = {}".format(i, mass, self.positions[body], volume_scaling))
                        new_lumen_volume += volume_scaling * mass

            ###############################################################################################################

            ###############################################################################################################
            # IF THERE IS AN EXISTING LUMEN THAT WE ARE ADDING TO
            else:
                print("CURRENTLY THERE IS A LUMEN")
                print("CURRENT POSITIONS {}".format(self.positions))
                print("THERE IS ALREADY A LUMEN AT POSITION {}. THE BODIES IN CENTRE ARE {}".format(self.N_bodies-1, bodies_in_centre))
                print("THERE ARE {} BODIES IN THE CENTRE".format(len(bodies_in_centre)))

                new_N_bodies = new_cell_number + 1 # remaining cells + lumen
                new_positions = np.zeros((new_N_bodies, 3))
                new_ages = np.zeros((new_N_bodies, 1))
                new_lifetimes = np.zeros((new_N_bodies, 1))
                print("remaining cells are {} and inner cells (+lumen) are {}".format(new_exterior_cells, bodies_in_centre))      
                ################## Calculate the new lumen volume##################
                # Existing lumen contributes all of its lumen, and additional cells contribute (volume_scaling * cell_volume)
                #TO DO -> Change mass to volume for clarity
                for i, body in enumerate(bodies_in_centre):
                    if body != self.N_bodies-1:
                        mass = self.r_min * (4/3) * np.pi * (1+ ((np.cbrt(2)-1)*self.ages[body,0]/self.lifetimes[body,0]))**3
                        added_mass = mass * volume_scaling
                        new_lumen_volume += added_mass
                        print("Adding a new cell, i,body = {},{} with age {} position {} and volume {} x scaling {}".format(i, body, ages[body,0], self.positions[body], mass, volume_scaling ))
                    else:
                        mass = self.r_min * (4/3) * np.pi * (1+ ((np.cbrt(2)-1)*self.ages[-1,0]/mean_lifetime))**3
                        added_mass = mass
                        new_lumen_volume += added_mass #add the existing lumen volume
                        print("Adding the existing lumen, i,body = {},{} with age {} position {} and volume {} ".format(i, body, ages[body,0], self.positions[body], mass ))
    
            
            print("NOW CALCULATING THE POSITION OF THE NEW LUMEN")              
            for i, body in enumerate(new_exterior_cells):  
                mass = self.r_min * (4/3) * np.pi * (1+ ((np.cbrt(2)-1)*self.ages[body,0]/self.lifetimes[body]))**3
                numerator += mass * self.positions[body]
                denominator += mass
                print("Adding new exterior cell i={}, position = {}, mass ={}".format(body, self.positions[body], mass))

            ##################Reorder the Positions and Ages##################
            for count, cell_index in enumerate(new_exterior_cells):
                new_positions[count] = self.positions[cell_index]
                new_ages[count] = self.ages[cell_index]
                new_lifetimes[count] = self.lifetimes[cell_index]

            lumen_scale_age = (mean_lifetime/(np.cbrt(2) -1)) *(((1/self.r_min) * np.cbrt(3*new_lumen_volume * 0.25 * (1/np.pi)))-1)
            self.lumen_radius = (0.75 * new_lumen_volume * (1/np.pi))**(1/3)
            self.lumen_volume = new_lumen_volume
            
            new_ages[-1] = lumen_scale_age 
            new_lifetimes[-1] = mean_lifetime
            new_positions[-1,:] = numerator/denominator 
            print("NEW POSITION OF LUMEN + {}".format(new_positions)) 
            self.ages = new_ages
            self.lifetimes = new_lifetimes
            self.N_bodies = self.ages.shape[0]
            self.positions = new_positions
            self.lumen_status = True
        
            try:
                print("DONE LUMEN JOINING, TRY JOIN LUMEN")
                self.hull=sp.ConvexHull(self.positions)
                print("SUCCESSFUL HULL ATTEMPT")
                self.last_event_time = all_t_values[-1]
                self.all_lumen_volumes.append(self.lumen_volume)

                print("THE LUMEN NOW HAS  AGE{} and volume {}".format(lumen_scale_age, self.lumen_volume))    
                print("AFTER LUMEN JOINING THERE ARE {} BODIES AND THE LAST EVENT TIME IS {}".format(N_bodies, self.last_event_time))
                print("AFTER LUMEN JOINING THE AGES ARE", self.ages)
                print("AFTER LUMEN JOINING THE LIFETIMES ARE", self.lifetimes)
            except:
                print("HULL FAILED, positions = {}, ages ={}".format(self.positions, self.ages))
                self.fail_cause = "all_lumen"

        else:
            print("TOO FEW CELLS TO ATTEMPT LUMEN FORMATION")
            self.fail_cause = "all_lumen"

    def perform_divisions(
        self
    ): 
        print("PERFORM DIVISIONS TRIGGERED AT TIME {} WHEN THERE ARE {} BODIES INITIALLY".format(all_t_values[-1], N_bodies))
        # print("WHEN DIVISION IS TRIGGERED THE AGES AND LIFETIMES ARE {}".format([(ages[i,0], lifetimes[i,0]) for i in range(N_bodies)]))
        N_old = self.N_bodies
        old_ages = self.ages
        old_positions = self.positions
        ################## Find out how many cells need to divide ##################
        # If there is no lumen, any age> lifetime denotes an old cell
        if self.lumen_status == False:
            number_dividing = self.ages[self.ages > self.lifetimes].size
            enumerator = self.ages
            start_index = 0
        # If there is a lumen, its age can exceed the lifetime
        else:
            number_dividing = self.ages[:-1][self.ages[:-1] > self.lifetimes[:-1]].size
            enumerator = self.ages[:-1]
            start_index = -1
        # print("THERE ARE {} CELLS DIVIDING".format(number_dividing))
        
        self.N_bodies =  N_old + number_dividing
        new_positions = np.zeros((self.N_bodies,3))
        new_ages = np.zeros((self.N_bodies, 1))
        new_lifetimes = np.zeros((self.N_bodies, 1))
        j = 0 #keep track of the number of dividing cells we've dealt with 
        for i, age in enumerate(enumerator): ##############CHECKKKKKKKKKKKKKKK
            if age < self.lifetimes[i]:
                new_positions[i] = self.positions[i]
                new_ages[i] = self.ages[i]
                new_lifetimes[i] = self.lifetimes[i]
            else:
                new_ages[i] = 0
                new_lifetimes[i] = np.random.normal(mean_lifetime, lifetime_std)
                new_ages[N_old + j+ start_index] = 0
                new_lifetimes[N_old + j + start_index] = np.random.normal(mean_lifetime, lifetime_std)
                #print("N_old, j =", N_old, j, "new ages [N_old + j]", new_ages[N_old + j], "new_lifetimes[N_old + j]",new_lifetimes[N_old + j] )
                theta_rand = np.random.uniform(0,360)
                phi_rand = np.random.uniform(0,360)

                # print("CURRENT NEW LIFETIMES ARE", new_lifetimes)
                new_positions[i][0] = self.positions[i][0] + self.r_min*np.cos(phi_rand)* np.sin(theta_rand)
                new_positions[i][1] = self.positions[i][1] +  self.r_min * np.sin(phi_rand) * np.sin(theta_rand)
                new_positions[i][2] = self.positions[i][2] +  self.r_min * np.cos(theta_rand)
                new_positions[N_old + j + start_index][0] =  new_positions[i][0]-(2*self.r_min*np.cos(phi_rand)* np.sin(theta_rand))
                new_positions[N_old + j + start_index][1] = new_positions[i][1]-(2*self.r_min * np.sin(phi_rand) * np.sin(theta_rand))
                new_positions[N_old + j+ start_index][2] = new_positions[i][2] - (2*self.r_min * np.cos(theta_rand))        
                j += 1

        if self.lumen_status == True:
            # If there's a lumen, its age and position should stay at position [-1]
            new_ages[-1] = old_ages[-1]
            new_lifetimes[-1] = mean_lifetime
            new_positions[-1] = old_positions[-1]
        self.positions = new_positions
        self.ages = new_ages
        self.lifetimes = new_lifetimes
        self.hull = sp.ConvexHull(self.positions)
        self.N_bodies = self.ages.shape[0]
        self.last_event_time = all_t_values[-1]
        print("THE AGES ARE NOW", self.ages)
        # print("AFTER DIVISION THERE ARE {} BODIES AND THE LAST EVENT TIME IS {}".format(N_bodies, last_event_time))
        # print("AFTER DIVISION THE AGES ARE", ages)
        # print("AFTER DIVISION THE LIFETIMES ARE", lifetimes)
        # print("LUMEN STATUS =", lumen_status)

    def execute(
        self,
        delta_t_max=0.001,
        N_bodies=4,
        boxsize=5,
        r_min,
        beta,
        alpha,
        A_eq_star_scaling,
        P_star,
        mean_lifetime,
        volume_scaling,
    ):
        """
        """

        self.r_min = r_min
        self.beta = beta
        self.alpha = alpha
        self.A_eq_star_scaling = A_eq_star_scaling
        self.P_star = P_star
        self.mean_lifetime = mean_lifetime
        self.volume_scaling = volume_scaling
        self.delta_t_max = delta_t_max
        self.N_bodies = N_bodies
        self.boxsize = boxsize

        t_min = 0
        while t_min < t_max:
            try:
                sol = solve_ivp(r_dot, (t_min, t_max), positions.flatten(), method='RK45', first_step=0.0003, max_step=delta_t_max, events=(events))
                t_min = max(sol.t)
                if t_min > t_max:
                    self.fail_cause = "success"
                    break
                if any (current_areas) <  A_eq_star:
                    self.fail_cause = "small_area"
                    print("BROKE DUE TO SMALL AREAS")
                    break
                if (self.lumen_status == True) & (positions.shape[0]<=3):
                    self.fail_cause = "all_lumen"
                    break
                ###################### CURRENTLY NO LUMEN ######################
                if self.lumen_status == False:
                    if len(self.hull.vertices) != self.N_bodies:
                        print("THERE ARE {} BODIES AND THE VERTICES ARE {}".format(N_bodies, hull.vertices))
                        cell_joins_lumen()
                        if self.fail_cause == "all_lumen":
                            break
                    elif any(self.ages >= lifetimes):   
                        perform_divisions()
                ###################### THERE IS AN EXISTING LUMEN ######################
                else:
                    if len([i for i in range(N_bodies) if np.linalg.norm(positions - positions[-1], axis=1) [i] <= (self.lumen_radius*radius_scaling)]) > 1: # if any of the bodies has a distance that's too close to the lumen
                        cell_joins_lumen()
                        if self.fail_cause == "all_lumen":
                            break
                    elif any(self.ages[0:-1] > lifetimes[0:-1]):
                        perform_divisions()
                    else:
                        self.fail_cause = "unknown"
                        print("ISSUE HERE")
            except:
                print("fail")
                print("positions = ", positions)
                break
        del self.all_t_values[0:2]

        self.results = pd.DataFrame()
        self.results = pd.DataFrame(list(zip(*[self.all_t_values, all_volumes, average_areas,  all_lumen_volumes, all_preferred_areas])))
        self.results.columns = ["t", "Cluster_Vol", "Areas", "Lumen_Volume", "Preferred_area"]
        self.results["Run_No"] = run_number

        self.results["r_min"] = r_min
        self.results["beta"] = beta
        self.results["alpha"] = alpha
        self.results["A_eq_star_scaling"] = A_eq_star_scaling
        self.results["P_star"] = P_star
        self.results["mean_lifetime"] = mean_lifetime
        self.results["lumen_volume_scaling"] = volume_scaling
        self.results["lumen_radius_scaling"] = radius_scaling
        self.results["end_reason"] = fail_cause
        
        df.to_parquet(".\\Animating_Output\\BM_based_on_area\\Alter{}_Run{}.parquet".format(alter, run_number))
        np.save('.\\Animating_Output\\BM_based_on_area\\Alter{}_positions_{}.npy'.format(alter, run_number), np.array(all_positions, dtype=object), allow_pickle=True)
        np.save('.\\Animating_Output\\BM_based_on_area\\Alter{}_ages_{}.npy'.format(alter, run_number), np.array(all_ages, dtype=object), allow_pickle=True)