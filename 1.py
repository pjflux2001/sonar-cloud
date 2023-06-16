# %% [markdown]
# Outline
# 
#     load data
#     initialization
#         load 1D array of data (X, Y, Z, W)
#         create 1D array of agents (X, Y, Z, polar angle, azimuth, W)
#         create 3D trace grid (empty)
#         create 3D dual deposit grid (empty)
#     simulation
#         update state
#         draw GUI
#         process data (kernel inputs: data, deposit)
#         propagation step (kernel inputs: agents, trace, deposit)
#         relaxation step (kernel inputs: deposit)
#         relaxation step (kernel inputs: trace)
#         generate visualization (kernel inputs: deposit, trace, vis buffer)
#         update GUI
#     store grids
#     tidy up
print('\n' * 10)
# %%
import numpy as np
import math, os
from enum import IntEnum
from numpy.random import default_rng
import time
from datetime import datetime
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage.filters import gaussian_filter
import taichi as ti
import taichi.math as timath

## Type aliases
FLOAT_CPU = np.float32
INT_CPU = np.int32
FLOAT_GPU = ti.f32
INT_GPU = ti.i32

VEC2i = ti.types.vector(2, INT_GPU)
VEC3i = ti.types.vector(3, INT_GPU)
VEC4i = ti.types.vector(4, INT_GPU)
VEC2f = ti.types.vector(2, FLOAT_GPU)
VEC3f = ti.types.vector(3, FLOAT_GPU)
VEC4f = ti.types.vector(4, FLOAT_GPU)

## Distance sampling distribution for agents
class EnumDistanceSamplingDistribution(IntEnum):
    CONSTANT = 0
    EXPONENTIAL = 1
    MAXWELL_BOLTZMANN = 2

## Directional sampling distribution for agents
class EnumDirectionalSamplingDistribution(IntEnum):
    DISCRETE = 0
    CONE = 1

## Sampling strategy for directional agent mutation
class EnumDirectionalMutationType(IntEnum):
    DETERMINISTIC = 0
    PROBABILISTIC = 1

## Deposit fetching strategy
class EnumDepositFetchingStrategy(IntEnum):
    NN = 0
    NN_PERTURBED = 1

## Handling strategy for agents that leave domain boundary
class EnumAgentBoundaryHandling(IntEnum):
    WRAP = 0
    REINIT_CENTER = 1
    REINIT_RANDOMLY = 2

# %%
## Default root directory
ROOT = '../../'

## Data input file - leave empty for random set
# INPUT_FILE = ''
INPUT_FILE = ROOT + 'data/csv/sample_3D_linW.csv'

## Simulation-wide constants
N_DATA_DEFAULT = 1000
N_AGENTS_DEFAULT = 1000000
DOMAIN_SIZE_DEFAULT = 100.0
TRACE_RESOLUTION_MAX = 1024
DEPOSIT_DOWNSCALING_FACTOR = 1
STEERING_RATE = 0.5
MAX_DEPOSIT = 10.0
DOMAIN_MARGIN = 0.05
VIS_RESOLUTION = (1920, 1080)
RAY_EPSILON = 1.0e-3

## State flags
distance_sampling_distribution = EnumDistanceSamplingDistribution.MAXWELL_BOLTZMANN
directional_sampling_distribution = EnumDirectionalSamplingDistribution.CONE
directional_mutation_type = EnumDirectionalMutationType.PROBABILISTIC
deposit_fetching_strategy = EnumDepositFetchingStrategy.NN_PERTURBED
agent_boundary_handling = EnumAgentBoundaryHandling.REINIT_RANDOMLY

# %%
## Initialize Taichi
ti.init(arch=ti.vulkan, device_memory_GB=4.0, debug=True)
rng = default_rng()

## Initialize data and agents
data = None
DOMAIN_MIN = None
DOMAIN_MAX = None
DOMAIN_SIZE = None
DOMAIN_CENTER = None
N_DATA = None
N_AGENTS = N_AGENTS_DEFAULT
AVG_WEIGHT = 10.0

## Load data
## If no input file then generate a random dataset
if len(INPUT_FILE) > 0:
    data = np.loadtxt(INPUT_FILE, delimiter=",").astype(FLOAT_CPU)
    N_DATA = data.shape[0]
    domain_min = (np.min(data[:,0]), np.min(data[:,1]), np.min(data[:,2]))
    domain_max = (np.max(data[:,0]), np.max(data[:,1]), np.max(data[:,2]))
    domain_size = np.subtract(domain_max, domain_min)
    DOMAIN_MIN = (domain_min[0] - DOMAIN_MARGIN * domain_size[0], domain_min[1] - DOMAIN_MARGIN * domain_size[1], domain_min[2] - DOMAIN_MARGIN * domain_size[2])
    DOMAIN_MAX = (domain_max[0] + DOMAIN_MARGIN * domain_size[0], domain_max[1] + DOMAIN_MARGIN * domain_size[1], domain_max[2] + DOMAIN_MARGIN * domain_size[2])
    DOMAIN_SIZE = np.subtract(DOMAIN_MAX, DOMAIN_MIN)
    AVG_WEIGHT = np.mean(data[:,3])
else:
    N_DATA = N_DATA_DEFAULT
    N_AGENTS = N_AGENTS_DEFAULT
    DOMAIN_SIZE = (DOMAIN_SIZE_DEFAULT, DOMAIN_SIZE_DEFAULT, DOMAIN_SIZE_DEFAULT)
    DOMAIN_MIN = (0.0, 0.0, 0.0)
    DOMAIN_MAX = DOMAIN_SIZE
    data = np.zeros(shape=(N_DATA, 4), dtype = FLOAT_CPU)
    data[:, 0] = rng.normal(loc = DOMAIN_MIN[0] + 0.5 * DOMAIN_MAX[0], scale = 0.15 * DOMAIN_SIZE[0], size = N_DATA)
    data[:, 1] = rng.normal(loc = DOMAIN_MIN[1] + 0.5 * DOMAIN_MAX[1], scale = 0.15 * DOMAIN_SIZE[1], size = N_DATA)
    data[:, 2] = rng.normal(loc = DOMAIN_MIN[2] + 0.5 * DOMAIN_MAX[2], scale = 0.15 * DOMAIN_SIZE[2], size = N_DATA)
    data[:, 3] = AVG_WEIGHT

## Derived constants
DOMAIN_CENTER = (0.5 * (DOMAIN_MIN[0] + DOMAIN_MAX[0]), 0.5 * (DOMAIN_MIN[1] + DOMAIN_MAX[1]), 0.5 * (DOMAIN_MIN[2] + DOMAIN_MAX[2]))
DATA_TO_AGENTS_RATIO = FLOAT_CPU(N_DATA) / FLOAT_CPU(N_AGENTS)
DOMAIN_SIZE_MAX = np.max([DOMAIN_SIZE[0], DOMAIN_SIZE[1], DOMAIN_SIZE[2]])
TRACE_RESOLUTION = INT_CPU((\
    FLOAT_CPU(TRACE_RESOLUTION_MAX) * DOMAIN_SIZE[0] / DOMAIN_SIZE_MAX,\
    FLOAT_CPU(TRACE_RESOLUTION_MAX) * DOMAIN_SIZE[1] / DOMAIN_SIZE_MAX,\
    FLOAT_CPU(TRACE_RESOLUTION_MAX) * DOMAIN_SIZE[2] / DOMAIN_SIZE_MAX))
DEPOSIT_RESOLUTION = (TRACE_RESOLUTION[0] // DEPOSIT_DOWNSCALING_FACTOR, TRACE_RESOLUTION[1] // DEPOSIT_DOWNSCALING_FACTOR, TRACE_RESOLUTION[2] // DEPOSIT_DOWNSCALING_FACTOR)

## Init agents
agents = np.zeros(shape=(N_AGENTS, 6), dtype = FLOAT_CPU)
agents[:, 0] = rng.uniform(low = DOMAIN_MIN[0] + 0.001, high = DOMAIN_MAX[0] - 0.001, size = N_AGENTS)
agents[:, 1] = rng.uniform(low = DOMAIN_MIN[1] + 0.001, high = DOMAIN_MAX[1] - 0.001, size = N_AGENTS)
agents[:, 2] = rng.uniform(low = DOMAIN_MIN[2] + 0.001, high = DOMAIN_MAX[2] - 0.001, size = N_AGENTS)
agents[:, 3] = np.arccos(2.0 * np.array(rng.uniform(low = 0.0, high = 1.0, size = N_AGENTS)) - 1.0) # zenith angle
agents[:, 4] = rng.uniform(low = 0.0, high = 2.0 * np.pi, size = N_AGENTS) # azimuth angle
agents[:, 5] = 1.0

print('Simulation domain min:', DOMAIN_MIN)
print('Simulation domain max:', DOMAIN_MAX)
print('Simulation domain size:', DOMAIN_SIZE)
print('Trace grid resolution:', TRACE_RESOLUTION)
print('Deposit grid resolution:', DEPOSIT_RESOLUTION)
print('Vis resolution:', VIS_RESOLUTION)
print('Data sample:', data[0, :])
print('Agent sample:', agents[0, :])
print('Number of agents:', N_AGENTS)
print('Number of data points:', N_DATA)

# %%
## Allocate GPU memory fields
## Keep in mind that the dimensions of these fields are important in the subsequent computations;
## that means if they change the GPU kernels and the associated handling code must be modified as well
data_field = ti.Vector.field(n = 4, dtype = FLOAT_GPU, shape = N_DATA)
agents_field = ti.Vector.field(n = 6, dtype = FLOAT_GPU, shape = N_AGENTS)
deposit_field = ti.Vector.field(n = 2, dtype = FLOAT_GPU, shape = DEPOSIT_RESOLUTION)
trace_field = ti.Vector.field(n = 1, dtype = FLOAT_GPU, shape = TRACE_RESOLUTION)
vis_field = ti.Vector.field(n = 3, dtype = FLOAT_GPU, shape = VIS_RESOLUTION)
print('Total GPU memory allocated:', INT_CPU(4 * (\
    data_field.shape[0] * 4 + \
    agents_field.shape[0] * 6 + \
    deposit_field.shape[0] * deposit_field.shape[1] * deposit_field.shape[2] * 2 + \
    trace_field.shape[0] * trace_field.shape[1] * trace_field.shape[2] * 1 + \
    vis_field.shape[0] * vis_field.shape[1] * 3 \
    ) / 2 ** 20), 'MB')

# %%
## Define all GPU functions and kernels
@ti.kernel
def zero_field(f: ti.template()):
    for cell in ti.grouped(f):
        f[cell].fill(0.0)
    return

@ti.kernel
def copy_field(dst: ti.template(), src: ti.template()): 
    for cell in ti.grouped(dst):
        dst[cell] = src[cell]
    return

@ti.func
def world_to_grid_3D(pos_world, domain_min, domain_max, grid_resolution) -> VEC3i:
    pos_relative = (pos_world - domain_min) / (domain_max - domain_min)
    grid_coord = ti.cast(pos_relative * ti.cast(grid_resolution, FLOAT_GPU), INT_GPU)
    return ti.max(VEC3i(0, 0, 0), ti.min(grid_coord, grid_resolution - (1, 1, 1)))

@ti.func
def angles_to_dir_3D(theta, phi) -> VEC3f:
    return timath.normalize(VEC3f(ti.sin(theta) * ti.cos(phi), ti.cos(theta), ti.sin(theta) * ti.sin(phi)))

@ti.func
def dir_3D_to_angles(dir):
    theta = timath.acos(dir[1] / timath.length(dir))
    phi = timath.atan2(dir[2], dir[0])
    return theta, phi

@ti.func
def axial_rotate_3D(vec, axis, angle):
    return ti.cos(angle) * vec + ti.sin(angle) * (timath.cross(axis, vec)) + timath.dot(axis, vec) * (1.0 - ti.cos(angle)) * axis

@ti.func
def custom_mod(a, b) -> FLOAT_GPU:
    return a - b * ti.floor(a / b)

@ti.kernel
def data_step(data_deposit: FLOAT_GPU, current_deposit_index: INT_GPU):
    for point in ti.ndrange(data_field.shape[0]):
        pos = VEC3f(0.0, 0.0, 0.0)
        pos[0], pos[1], pos[2], weight = data_field[point]
        deposit_cell = world_to_grid_3D(pos, VEC3f(DOMAIN_MIN), VEC3f(DOMAIN_MAX), VEC3i(DEPOSIT_RESOLUTION))
        deposit_field[deposit_cell][current_deposit_index] += data_deposit * weight
    return

@ti.kernel
def agent_step(sense_distance: FLOAT_GPU,\
               sense_angle: FLOAT_GPU,\
               steering_rate: FLOAT_GPU,\
               sampling_exponent: FLOAT_GPU,\
               step_size: FLOAT_GPU,\
               agent_deposit: FLOAT_GPU,\
               current_deposit_index: INT_GPU,\
               distance_sampling_distribution: INT_GPU,\
               directional_sampling_distribution: INT_GPU,\
               directional_mutation_type: INT_GPU,\
               deposit_fetching_strategy: INT_GPU,\
               agent_boundary_handling: INT_GPU):
    for agent in ti.ndrange(agents_field.shape[0]):
        pos = VEC3f(0.0, 0.0, 0.0)
        pos[0], pos[1], pos[2], theta, phi, weight = agents_field[agent]

        ## Generate sensing distance for the agent, constant or probabilistic
        agent_sensing_distance = sense_distance
        distance_scaling_factor = 1.0
        if distance_sampling_distribution == EnumDistanceSamplingDistribution.EXPONENTIAL:
            xi = timath.clamp(ti.random(dtype=FLOAT_GPU), 0.001, 0.999) ## log & pow are unstable in extremes
            distance_scaling_factor = -ti.log(xi)
        elif distance_sampling_distribution == EnumDistanceSamplingDistribution.MAXWELL_BOLTZMANN:
            xi = timath.clamp(ti.random(dtype=FLOAT_GPU), 0.001, 0.999) ## log & pow are unstable in extremes
            distance_scaling_factor = -0.3033 * ti.log( (ti.pow(xi + 0.005, -0.4) - 0.9974) / 7.326 )
        agent_sensing_distance *= distance_scaling_factor

        dir_fwd = angles_to_dir_3D(theta, phi)
        xi_dir = 1.0
        if directional_sampling_distribution == EnumDirectionalSamplingDistribution.CONE:
            xi_dir = ti.random(dtype=FLOAT_GPU)
        theta_sense = theta - xi_dir * sense_angle
        off_fwd_dir = angles_to_dir_3D(theta_sense, phi)
        random_azimuth = ti.random(dtype=FLOAT_GPU) * 2.0 * timath.pi - timath.pi
        dir_mut = axial_rotate_3D(off_fwd_dir, dir_fwd, random_azimuth)

        deposit_fwd = deposit_field[world_to_grid_3D(pos + agent_sensing_distance * dir_fwd, VEC3f(DOMAIN_MIN), VEC3f(DOMAIN_MAX), VEC3i(DEPOSIT_RESOLUTION))][current_deposit_index]
        deposit_mut = deposit_field[world_to_grid_3D(pos + agent_sensing_distance * dir_mut, VEC3f(DOMAIN_MIN), VEC3f(DOMAIN_MAX), VEC3i(DEPOSIT_RESOLUTION))][current_deposit_index]

        p_remain = ti.pow(deposit_fwd, sampling_exponent)
        p_mutate = ti.pow(deposit_mut, sampling_exponent)
        dir_new = dir_fwd
        theta_new = theta
        phi_new = phi
        if p_remain + p_mutate > 1.0e-5:
            if ti.random(dtype=FLOAT_GPU) < (p_mutate / (p_remain + p_mutate)):
                theta_mut = theta - steering_rate * xi_dir * sense_angle
                off_mut_dir = angles_to_dir_3D(theta_mut, phi)
                dir_new = axial_rotate_3D(off_mut_dir, dir_fwd, random_azimuth)
                theta_new, phi_new = dir_3D_to_angles(dir_new)

        ## Generate new direction for the agent based on the sampled deposit
        pos_new = pos + step_size * distance_scaling_factor * dir_new

        ## Agent behavior at domain boundaries
        if agent_boundary_handling == EnumAgentBoundaryHandling.WRAP:
            pos_new[0] = custom_mod(pos_new[0] - DOMAIN_MIN[0] + DOMAIN_SIZE[0], DOMAIN_SIZE[0]) + DOMAIN_MIN[0]
            pos_new[1] = custom_mod(pos_new[1] - DOMAIN_MIN[1] + DOMAIN_SIZE[1], DOMAIN_SIZE[1]) + DOMAIN_MIN[1]
            pos_new[2] = custom_mod(pos_new[2] - DOMAIN_MIN[2] + DOMAIN_SIZE[2], DOMAIN_SIZE[2]) + DOMAIN_MIN[2]
        elif agent_boundary_handling == EnumAgentBoundaryHandling.REINIT_CENTER:
            # if pos_new[0] <= DOMAIN_MIN[0] or pos_new[0] >= DOMAIN_MAX[0] or pos_new[1] <= DOMAIN_MIN[1] or pos_new[1] >= DOMAIN_MAX[1] or pos_new[2] <= DOMAIN_MIN[2] or pos_new[2] >= DOMAIN_MAX[2]:
            #     pos_new[0] = 0.5 * (DOMAIN_MIN[0] + DOMAIN_MAX[0])
            #     pos_new[1] = 0.5 * (DOMAIN_MIN[1] + DOMAIN_MAX[1])
            #     pos_new[2] = 0.5 * (DOMAIN_MIN[2] + DOMAIN_MAX[2])
            # TODO - temporary solution for agent reinit at data:
            if pos_new[0] <= DOMAIN_MIN[0] or pos_new[0] >= DOMAIN_MAX[0] or pos_new[1] <= DOMAIN_MIN[1] or pos_new[1] >= DOMAIN_MAX[1] or pos_new[2] <= DOMAIN_MIN[2] or pos_new[2] >= DOMAIN_MAX[2]:
                random_data_index = ti.random(dtype=INT_GPU) % N_DATA
                pos_new[0] = data_field[random_data_index][0]
                pos_new[1] = data_field[random_data_index][1]
                pos_new[2] = data_field[random_data_index][2]
        elif agent_boundary_handling == EnumAgentBoundaryHandling.REINIT_RANDOMLY:
            if pos_new[0] <= DOMAIN_MIN[0] or pos_new[0] >= DOMAIN_MAX[0] or pos_new[1] <= DOMAIN_MIN[1] or pos_new[1] >= DOMAIN_MAX[1] or pos_new[2] <= DOMAIN_MIN[2] or pos_new[2] >= DOMAIN_MAX[2]:
                pos_new[0] = DOMAIN_MIN[0] + timath.clamp(ti.random(dtype=FLOAT_GPU), 0.001, 0.999) * DOMAIN_SIZE[0]
                pos_new[1] = DOMAIN_MIN[1] + timath.clamp(ti.random(dtype=FLOAT_GPU), 0.001, 0.999) * DOMAIN_SIZE[1]
                pos_new[2] = DOMAIN_MIN[2] + timath.clamp(ti.random(dtype=FLOAT_GPU), 0.001, 0.999) * DOMAIN_SIZE[2]

        agents_field[agent][0] = pos_new[0]
        agents_field[agent][1] = pos_new[1]
        agents_field[agent][2] = pos_new[2]
        agents_field[agent][3] = theta_new
        agents_field[agent][4] = phi_new

        ## Generate deposit and trace at the new position
        deposit_cell = world_to_grid_3D(pos_new, VEC3f(DOMAIN_MIN), VEC3f(DOMAIN_MAX), VEC3i(DEPOSIT_RESOLUTION))
        deposit_field[deposit_cell][current_deposit_index] += agent_deposit * weight

        trace_cell = world_to_grid_3D(pos_new, VEC3f(DOMAIN_MIN), VEC3f(DOMAIN_MAX), VEC3i(TRACE_RESOLUTION))
        trace_field[trace_cell][0] += weight
    return

DIFFUSION_KERNEL = [1.0, 1.0, 0.0, 0.0]
DIFFUSION_KERNEL_NORM = DIFFUSION_KERNEL[0] + 6.0 * DIFFUSION_KERNEL[1] + 12.0 * DIFFUSION_KERNEL[2] + 8.0 * DIFFUSION_KERNEL[3]

@ti.kernel
def deposit_relaxation_step(attenuation: FLOAT_GPU, current_deposit_index: INT_GPU):
    for cell in ti.grouped(deposit_field):
        ## The "beautiful" expression below implements a 3x3x3 kernel diffusion in a 6-neighborhood with manually wrapped addressing
        ## Taichi doesn't support modulo for tuples so each dimension is handled separately
        value = DIFFUSION_KERNEL[0] * deposit_field[( (cell[0] + 0 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 0 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1], (cell[2] + 0 + DEPOSIT_RESOLUTION[2]) % DEPOSIT_RESOLUTION[2])][current_deposit_index]\
              + DIFFUSION_KERNEL[1] * deposit_field[( (cell[0] + 1 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 0 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1], (cell[2] + 0 + DEPOSIT_RESOLUTION[2]) % DEPOSIT_RESOLUTION[2])][current_deposit_index]\
              + DIFFUSION_KERNEL[1] * deposit_field[( (cell[0] - 1 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 0 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1], (cell[2] + 0 + DEPOSIT_RESOLUTION[2]) % DEPOSIT_RESOLUTION[2])][current_deposit_index]\
              + DIFFUSION_KERNEL[1] * deposit_field[( (cell[0] + 0 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 1 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1], (cell[2] + 0 + DEPOSIT_RESOLUTION[2]) % DEPOSIT_RESOLUTION[2])][current_deposit_index]\
              + DIFFUSION_KERNEL[1] * deposit_field[( (cell[0] + 0 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] - 1 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1], (cell[2] + 0 + DEPOSIT_RESOLUTION[2]) % DEPOSIT_RESOLUTION[2])][current_deposit_index]\
              + DIFFUSION_KERNEL[1] * deposit_field[( (cell[0] + 0 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 0 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1], (cell[2] + 1 + DEPOSIT_RESOLUTION[2]) % DEPOSIT_RESOLUTION[2])][current_deposit_index]\
              + DIFFUSION_KERNEL[1] * deposit_field[( (cell[0] + 0 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 0 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1], (cell[2] - 1 + DEPOSIT_RESOLUTION[2]) % DEPOSIT_RESOLUTION[2])][current_deposit_index]
        deposit_field[cell][1 - current_deposit_index] = attenuation * value / DIFFUSION_KERNEL_NORM
    return

@ti.kernel
def trace_relaxation_step(attenuation: FLOAT_GPU):
    for cell in ti.grouped(trace_field):
        ## Perturb the attenuation by a small factor to avoid accumulating quantization errors
        trace_field[cell][0] *= attenuation - 0.001 + 0.002 * ti.random(dtype=FLOAT_GPU)
    return

###
### PATH TRACER CODE STARTS HERE
###

@ti.func
def ray_AABB_intersection(ray_pos, ray_dir, AABB_min, AABB_max):
    t0 = (AABB_min[0] - ray_pos[0]) / ray_dir[0]
    t1 = (AABB_max[0] - ray_pos[0]) / ray_dir[0]
    t2 = (AABB_min[1] - ray_pos[1]) / ray_dir[1]
    t3 = (AABB_max[1] - ray_pos[1]) / ray_dir[1]
    t4 = (AABB_min[2] - ray_pos[2]) / ray_dir[2]
    t5 = (AABB_max[2] - ray_pos[2]) / ray_dir[2]
    t6 = ti.max(ti.max(ti.min(t0, t1), ti.min(t2, t3)), ti.min(t4, t5))
    t7 = ti.min(ti.min(ti.max(t0, t1), ti.max(t2, t3)), ti.max(t4, t5))
    return VEC2f(-1.0, -1.0) if (t7 < 0.0 or t6 >= t7) else VEC2f(t6, t7)

@ti.func
def tiutils_ray_aabb_intersection(o, d, box_min, box_max):
    intersect = 1

    near_int = -1e10
    far_int = 1e10

    for i in ti.static(range(3)):
        if d[i] == 0:
            if o[i] < box_min[i] or o[i] > box_max[i]:
                intersect = 0
        else:
            i1 = (box_min[i] - o[i]) / d[i]
            i2 = (box_max[i] - o[i]) / d[i]

            new_far_int = ti.max(i1, i2)
            new_near_int = ti.min(i1, i2)

            far_int = ti.min(new_far_int, far_int)
            near_int = ti.max(new_near_int, near_int)

    if near_int > far_int:
        intersect = 0
    return intersect, near_int, far_int

@ti.func
def shitty_colorizer(path_L):
    colorizer = path_L[0] + path_L[1] + path_L[2]
    if (colorizer > 150):
        # more red
        path_L[0] *= 1.5
        path_L[1] *= 0.5
        path_L[2] *= 0.5
    elif (colorizer > 125):
        # more purple
        path_L[0] *= 1.5
        path_L[1] *= 0.5
        path_L[2] *= 1.5
    elif (colorizer > 100):
        # more blue
        path_L[0] *= 0.5
        path_L[1] *= 0.5
        path_L[2] *= 1.5
    elif (colorizer > 75):
        # more cyan
        path_L[0] *= 0.5
        path_L[1] *= 1.5
        path_L[2] *= 1.5
    elif (colorizer > 50):
        # more green
        path_L[0] *= 0.5
        path_L[1] *= 1.5
        path_L[2] *= 0.5
    elif (colorizer > 25):
        # more yellow
        path_L[0] *= 1.5
        path_L[1] *= 1.5
        path_L[2] *= 0.5
    else:
        # more red
        path_L[0] *= 1.5
        path_L[1] *= 0.5
        path_L[2] *= 0.5
    return path_L

@ti.func
def get_dir_1(dir):
	inv_norm = 1.0 / ti.sqrt(dir[0]*dir[0] + dir[2]*dir[2]);
	return VEC3f(dir[2] * inv_norm, 0.0, -dir[0] * inv_norm);

@ti.func
def get_dir_2(dir, v1):
	return timath.cross(dir, v1);

@ti.func
def sample_HG(v, g):
    xi = ti.random()
    cos_theta = 0.0
    if (ti.abs(g) > 1.e-3):
        sqr_term = (1.0 - g*g) / (1.0 - g + 2.0*g*xi)
        cos_theta = (1.0 + g*g - sqr_term*sqr_term) / (2.0 * abs(g))
    else:
        cos_theta = 1.0 - 2.0*xi;

    sin_theta = timath.sqrt(ti.max(0.0, 1.0 - cos_theta*cos_theta))
    phi = (timath.pi * 2) * ti.random()
    
    v1 = get_dir_1(v)
    v2 = get_dir_2(v, v1)
    
    return sin_theta * ti.cos(phi) * v1 + sin_theta * ti.sin(phi) * v2 + cos_theta * v

@ti.func
def sample_volume(pos, trace_vis=1, deposit_vis=0):
    deposit_val = deposit_field[world_to_grid_3D(pos, VEC3f(DOMAIN_MIN), VEC3f(DOMAIN_MAX), VEC3i(DEPOSIT_RESOLUTION))][current_deposit_index]
    trace_val = trace_field[world_to_grid_3D(pos, VEC3f(DOMAIN_MIN), VEC3f(DOMAIN_MAX), VEC3i(TRACE_RESOLUTION))][0]
    density = (trace_val * trace_vis + deposit_val * deposit_vis) * sigma_e / 10

    # return VEC3f(deposit_val, deposit_val, deposit_val)
    return VEC3f(trace_vis * trace_val, deposit_vis * deposit_val, density)
    # ^^  the blue is density, covering most of the screen. reduce it.

    # return VEC3f(trace_val, trace_val, trace_val)

@ti.func
def expo_dist(range):
    return -ti.log(ti.max(ti.random(dtype=FLOAT_GPU), 0.001)) * range

@ti.func
def trace_to_rho(trace):
    sample_weight = 0.01
    trim_density = 1.0e-5
    ambient_trace = 0.0
    return sample_weight * (ti.max(trace - trim_density, 0.0) + ambient_trace)

@ti.func
def get_rho(ray_pos):
    return (sample_volume(ray_pos)[0])

@ti.func
def delta_step(sigma_max_inv, xi):
    return -ti.log(ti.max(xi, 0.001)) * sigma_max_inv

@ti.func
def delta_tracking(ray_pos, ray_dir, t_min, t_max, rho_max_inv, sigma_a, sigma_s):
    sigma_max_inv = rho_max_inv / (sigma_a + sigma_s)
    t = t_min
    event_rho = 0.0
    
    while (t <= t_max and ti.random(dtype=FLOAT_GPU) > event_rho * rho_max_inv):
        t += delta_step(sigma_max_inv, ti.random(dtype=FLOAT_GPU))
        event_rho = get_rho(ray_pos + t * ray_dir)

    return t;

def occlusion_tracking(rp, rd, t_min, t_max, rho_max_inv, subsampling):
    sigma_max_inv = rho_max_inv / (sigma_a + sigma_s)
    t = t_min
    rho_sum = 0.0
    iSteps = 0
    while (t <= t_max):
        t += subsampling * delta_step(sigma_max_inv, rng.random_float())
        rho_sum += get_rho(rp + t * rd)
        iSteps += 1
    rho_sum /= float(iSteps)
    transmittance = timath.exp(-(sigma_s + sigma_a) * rho_sum * (t_max - t_min))

    return transmittance;


@ti.func
def random_in_unit_sphere():
    phi = ti.random() * 2 * 3.14159
    cos_theta = 2 * ti.random() - 1
    u = ti.random()

    theta = ti.acos(cos_theta)
    r = ti.pow(u, 1/3)

    x = r * ti.sin(theta) * ti.cos(phi)
    y = r * ti.sin(theta) * ti.sin(phi)
    z = r * ti.cos(theta)

    return VEC3f(x, y, z)

COLORMAP_RESOLUTION = [123, 5, 3]
img = ti.field(float, shape=COLORMAP_RESOLUTION)
img.from_numpy(ti.tools.imread('colormap.png'))

@ti.func
def TexSamplePosition(pos: FLOAT_GPU):
    # find an index
    xPos = ti.min(ti.floor(pos * COLORMAP_RESOLUTION[0], ti.i32), COLORMAP_RESOLUTION[0] - 1)
    red = img[xPos, 1, 0]
    green = img[xPos, 1, 1]
    blue = img[xPos, 1, 2]
    return [red, green, blue]


@ti.func
def get_emitted_trace_L(pos: FLOAT_GPU):
    return TexSamplePosition(pos)

@ti.func
def random_in_unit_sphere(range):
    phi = ti.random() * 2 * 3.14159
    cos_theta = 2 * ti.random() - 1
    u = ti.random()

    theta = ti.acos(cos_theta)
    r = ti.pow(u, 1/3)

    x = r * ti.sin(theta) * ti.cos(phi)
    y = r * ti.sin(theta) * ti.sin(phi)
    z = r * ti.cos(theta)

    return VEC3f(x, y, z)

# RENDERER SETTINGS
_CLEAR_COLOR = VEC3f(0.0, 0.0, 0.0)
max_bounces = 6
num_samples = 3
_USE_RUSSIAN_ROULETTE = False
_HISTOGRAM_BASE = 10.0
_RUSSIAN_ROULETTE_START_ORDER = 2
_DIRECTION_SAMPLE_RADIUS = 1
throughput_multiplier = 1.075
trace_max = 2.89
use_ray_marcher = False
use_debug_mode = False
scattering_anisotropy = 0.9

# ADD:
# - ray marcher fallback
# - proper throughput
# - 
@ti.kernel
def render_visualization_volumetric_pathtraced(camera_distance: FLOAT_GPU, camera_polar: FLOAT_GPU, camera_azimuth: FLOAT_GPU, accumulate_frame: INT_GPU, accumulation_count: INT_GPU, throughput_multiplier: FLOAT_GPU, num_samples: INT_GPU, max_bounces: INT_GPU, deposit_vis: FLOAT_GPU, trace_vis: FLOAT_GPU, scattering_anisotropy: FLOAT_GPU, sigma_e: FLOAT_GPU, sigma_a: FLOAT_GPU, sigma_s: FLOAT_GPU, sigma_t: FLOAT_GPU, trace_max: FLOAT_GPU, debug_mode: INT_GPU):
    ## INITIAL CONSTANTS
    aspect_ratio = ti.cast(VIS_RESOLUTION[0], FLOAT_GPU) / ti.cast(VIS_RESOLUTION[1], FLOAT_GPU) # the aspect ratio of the image
    screen_distance = DOMAIN_SIZE_MAX # the distance of the screen from the camera
    camera_offset = camera_distance * VEC3f(ti.cos(camera_azimuth) * ti.sin(camera_polar), ti.sin(camera_azimuth) * ti.sin(camera_polar), ti.cos(camera_polar)) # the offset of the camera from the center of the volume
    camera_pos = DOMAIN_CENTER + camera_offset # the position of the camera
    cam_Z = timath.normalize(-camera_offset) # the camera's z axis
    cam_Y = VEC3f(0.0, 0.0, 1.0)
    cam_X = timath.normalize(timath.cross(cam_Z, cam_Y)) # the camera's x axis
    cam_Y = timath.normalize(timath.cross(cam_X, cam_Z)) # the camera's y axis
    m_b = -999.0
    albedo = 0.0
    new_sigma_a = 0.0
    new_sigma_s = 0.0
    if (sigma_t > 1.e-5):
        albedo = sigma_s / sigma_t
        new_sigma_a = (1.0 - albedo) * sigma_t
        new_sigma_s = albedo * sigma_t
    for x, y in ti.ndrange(VIS_RESOLUTION[0], VIS_RESOLUTION[1]):
        ###
        ### FOR EACH PIXEL ...
        ###


        ## Compute x and y ray directions in neutral camera position
        rx = DOMAIN_SIZE_MAX * (ti.cast(x, FLOAT_GPU) / ti.cast(VIS_RESOLUTION[0], FLOAT_GPU)) - 0.5 * DOMAIN_SIZE_MAX # x coordinate of the ray in the camera's coordinate system
        ry = DOMAIN_SIZE_MAX * (ti.cast(y, FLOAT_GPU) / ti.cast(VIS_RESOLUTION[1], FLOAT_GPU)) - 0.5 * DOMAIN_SIZE_MAX # y coordinate of the ray in the camera's coordinate system
        ry /= aspect_ratio # correct for aspect ratio (+ resize handler, since this is computed per-frame)

        ## Initialize ray origin and direction
        screen_pos = camera_pos + rx * cam_X + ry * cam_Y + screen_distance * cam_Z # position of the ray in the world coordinate system

        # Great! Now we have everything we need for the ray for this pixel.
        rho_max_inv = 1.0 / trace_to_rho(trace_max);

        ## Get intersection of the ray with the volume AABB
        path_L = VEC3f(0.0, 0.0, 0.0) # the path radiance of the ray (total light it has accumulated)

        for s in range(num_samples):
            ray_pos = camera_pos # the position of the ray in the world coordinate system
            ray_dir = timath.normalize(screen_pos - ray_pos) # the direction of the ray in the world coordinate system

            t = ray_AABB_intersection(ray_pos, ray_dir, VEC3f(DOMAIN_MIN), VEC3f(DOMAIN_MAX)) # returns [near, far] of volume intersection

            if (t.y >= 0):
                t.x += RAY_EPSILON
                t.y -= RAY_EPSILON
                ray_pos += t.x * ray_dir # move the ray to the intersection point
                ray_dir = ray_dir * (t.y - t.x) # shorten the ray to the intersection interval
                ray_dir = timath.normalize(ray_dir) # normalize the ray direction
                t_event = 0.0
                throughput = 1.0

                for n in range(max_bounces):
                    t = ray_AABB_intersection(ray_pos, ray_dir, VEC3f(DOMAIN_MIN), VEC3f(DOMAIN_MAX)) # returns [near, far] of volume intersection
                    t_event = delta_tracking(ray_pos, ray_dir, 0.0, t.y, rho_max_inv, new_sigma_a, new_sigma_s)
                    if (t_event >= t.y):
                        break

                    m_b = sigma_t

                    ray_pos += t_event * ray_dir # move the ray to the intersection point

                    sample = sample_volume(ray_pos, trace_vis, deposit_vis)
                    # combine the trace and deposit values
                    rho_event = sample[0] + sample[1]

                    rho_event = rho_event * throughput_multiplier # sample the volume at the intersection point

                    emission = get_emitted_trace_L(rho_event)


                    if debug_mode == 1:
                        path_L = n
                    else:
                        path_L += throughput * rho_event * sigma_e * emission / 255

                    ray_dir = timath.normalize(sample_HG(ray_dir, scattering_anisotropy)) # sample HGF

                    if (_USE_RUSSIAN_ROULETTE == True):
                        if (n >= _RUSSIAN_ROULETTE_START_ORDER and ti.random() > albedo):
                            break
                        else:
                            throughput *= albedo
                    else:
                        throughput *= albedo


        # path_L = shitty_colorizer(path_L)
        

        if (accumulate_frame == True):
            new_value = path_L / num_samples
            old_value = vis_field[x, y]

            weight = 1.0 / (accumulation_count + 1)
            accumulatedPixel = old_value * (1 - weight) + new_value * weight
            vis_field[x, y] = accumulatedPixel # average the path radiance between this and the previous frame
        else:
            vis_field[x, y] += (path_L / num_samples) # average the path radiance over the number of samples
    print("aaa:", m_b)
    return


ray_marcher_steps = 200
@ti.kernel
def render_visualization_raymarched(throughput_multiplier: FLOAT_GPU, deposit_vis: FLOAT_GPU, trace_vis: FLOAT_GPU, camera_distance: FLOAT_GPU, camera_polar: FLOAT_GPU, camera_azimuth: FLOAT_GPU, n_steps_f: FLOAT_GPU, current_deposit_index: INT_GPU):
    n_steps = ti.cast(n_steps_f, INT_GPU)
    aspect_ratio = ti.cast(VIS_RESOLUTION[0], FLOAT_GPU) / ti.cast(VIS_RESOLUTION[1], FLOAT_GPU)
    screen_distance = DOMAIN_SIZE_MAX
    camera_offset = camera_distance * VEC3f(ti.cos(camera_azimuth) * ti.sin(camera_polar), ti.sin(camera_azimuth) * ti.sin(camera_polar), ti.cos(camera_polar))
    camera_pos = DOMAIN_CENTER + camera_offset
    cam_Z = timath.normalize(-camera_offset)
    cam_Y = VEC3f(0.0, 0.0, 1.0)
    cam_X = timath.normalize(timath.cross(cam_Z, cam_Y))
    cam_Y = timath.normalize(timath.cross(cam_X, cam_Z))

    for x, y in ti.ndrange(VIS_RESOLUTION[0], VIS_RESOLUTION[1]):
        ## Compute x and y ray directions in neutral camera position
        rx = DOMAIN_SIZE_MAX * (ti.cast(x, FLOAT_GPU) / ti.cast(VIS_RESOLUTION[0], FLOAT_GPU)) - 0.5 * DOMAIN_SIZE_MAX
        ry = DOMAIN_SIZE_MAX * (ti.cast(y, FLOAT_GPU) / ti.cast(VIS_RESOLUTION[1], FLOAT_GPU)) - 0.5 * DOMAIN_SIZE_MAX
        ry /= aspect_ratio

        ## Initialize ray origin and direction
        screen_pos = camera_pos + rx * cam_X + ry * cam_Y + screen_distance * cam_Z
        ray_dir = timath.normalize(screen_pos - camera_pos)

        ## Get intersection of the ray with the volume AABB
        t = ray_AABB_intersection(camera_pos, ray_dir, VEC3f(DOMAIN_MIN), VEC3f(DOMAIN_MAX))
        ray_L = VEC3f(0.0, 0.0, 0.0)
        ray_delta = 1.71 * DOMAIN_SIZE_MAX / n_steps_f

        ## Check if we intersect the volume AABB at all
        if t[1] >= 0.0:
            # Hit! Now we must find the color ...
            t[0] += RAY_EPSILON
            t[1] -= RAY_EPSILON
            t_current = t[0] + ti.random(dtype=FLOAT_GPU) * ray_delta
            ray_pos = camera_pos + t_current * ray_dir

            ## Main integration loop
            for i in ti.ndrange(n_steps):
                if t_current >= t[1]:
                    break
                ray_L += (sample_volume(ray_pos, trace_vis, deposit_vis) * throughput_multiplier) / n_steps_f
                ray_pos += ray_delta * ray_dir
                t_current += ray_delta

        color = shitty_colorizer(ray_L)
        vis_field[x, y] = color
    return
# %%
## Initialize GPU fields
data_field.from_numpy(data)
agents_field.from_numpy(agents)
zero_field(deposit_field)
zero_field(trace_field)
zero_field(vis_field)

# %%
## Main simulation & vis loop
sense_distance = 5.545 # 0.005 * DOMAIN_SIZE_MAX
sense_angle = 0.751
step_size = 0.00015 * DOMAIN_SIZE_MAX
sampling_exponent = 2.636
trace_vis = 0.5
deposit_vis = 0.5
deposit_attenuation = 0.85
trace_attenuation = 0.96
data_deposit = 0.1 * MAX_DEPOSIT
agent_deposit = 0#data_deposit * DATA_TO_AGENTS_RATIO
z_slice = 0.5

camera_distance = 3.0 * DOMAIN_SIZE_MAX
camera_polar = 0.5 * np.pi
camera_azimuth = 0.0
last_MMB = [-1.0, -1.0]
last_RMB = [-1.0, -1.0]

current_deposit_index = 0
do_simulate = True
do_render = True
hide_UI = False

window = ti.ui.Window('PolyPhy', (vis_field.shape[0], vis_field.shape[1]), show_window = True)
window.show()
canvas = window.get_canvas()

## Current timestamp
def stamp() -> str:
    return datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")

## Store current deposit and trace fields
def store_fit():
    if not os.path.exists(ROOT + "data/fits/"):
        os.makedirs(ROOT + "data/fits/")
    current_stamp = stamp()
    deposit = deposit_field.to_numpy()
    np.save(ROOT + 'data/fits/deposit_' + current_stamp + '.npy', deposit)
    trace = trace_field.to_numpy()
    np.save(ROOT + 'data/fits/trace_' + current_stamp + '.npy', trace)
    return current_stamp, deposit, trace

accumulation_count = 0

# MATERIAL SETTINGS [Adjusted in GUI]
sigma_s = 0.0
sigma_a = 0.5
sigma_e = 10.0
albedo = 0.5
sigma_t = sigma_s + sigma_a

def update_albedo_constraints(albedo, sigma_a, sigma_s, sigma_t):
    sigma_a = (1.0 - albedo) * sigma_t
    sigma_s = albedo * sigma_t

    return sigma_a, sigma_s

sigma_a, sigma_s = update_albedo_constraints(albedo, sigma_a, sigma_s, sigma_t)

## Main simulation and rendering loop
while window.running:
    do_export = False
    do_screenshot = False
    do_quit = False
    accumulate_frame = True
    

    ## Handle controls
    if window.get_event(ti.ui.PRESS):
        if window.event.key == 'e': do_export = True
        if window.event.key == 's': do_screenshot = True
        if window.event.key == 'h': hide_UI = not hide_UI
        if window.event.key in [ti.ui.ESCAPE]: do_quit = True
        if window.event.key in [ti.ui.UP]: camera_distance -= 0.25 * DOMAIN_SIZE_MAX
        if window.event.key in [ti.ui.DOWN]: camera_distance += 0.25 * DOMAIN_SIZE_MAX

    ## Handle camera control: rotation
    mouse_pos = window.get_cursor_pos()
    if window.is_pressed(ti.ui.RMB):
        accumulate_frame = False
        vis_field.fill(0.0)
        if last_RMB[0] < 0.0:
            last_RMB = mouse_pos
        else:
            delta_RMB = np.subtract(mouse_pos, last_RMB)
            last_RMB = mouse_pos
            camera_azimuth -= 5.0 * delta_RMB[0]
            camera_polar += 3.5 * delta_RMB[1]
            camera_polar = np.min([np.max([1.0e-2, camera_polar]), np.pi-1.0e-2])
    else:
        last_RMB = [-1.0, -1.0]
    
    ## Handle camera control: zooming
    if window.is_pressed(ti.ui.MMB):
        accumulate_frame = False
        vis_field.fill(0.0)
        if last_MMB[0] < 0.0:
            last_MMB = mouse_pos
        else:
            delta_MMB = np.subtract(mouse_pos, last_MMB)
            last_MMB = mouse_pos
            camera_distance -= 5.0 * DOMAIN_SIZE_MAX * delta_MMB[1]
            camera_distance = np.max([camera_distance, 0.85 * DOMAIN_SIZE_MAX])
    else:
        last_MMB = [-1.0, -1.0]

    if (accumulate_frame):
        accumulation_count += 1
    else:
        accumulation_count = 0

    if not hide_UI:
        ## Draw main interactive control GUI

        ###
        ### Simulation controls
        ###
        window.GUI.begin('Main', 0.01, 0.01, 0.32 * 1024.0 / FLOAT_CPU(VIS_RESOLUTION[0]), 0.85 * 1024.0 / FLOAT_CPU(VIS_RESOLUTION[1]))
        window.GUI.text("MCPM parameters:")
        sense_distance = window.GUI.slider_float('Sensing dist', sense_distance, 0.1, 0.05 * np.max([DOMAIN_SIZE[0], DOMAIN_SIZE[1]]))
        sense_angle = window.GUI.slider_float('Sensing angle', sense_angle, 0.01, 0.5 * np.pi)
        sampling_exponent = window.GUI.slider_float('Sampling expo', sampling_exponent, 1.0, 10.0)
        step_size = window.GUI.slider_float('Step size', step_size, 0.0, 0.0025 * np.max([DOMAIN_SIZE[0], DOMAIN_SIZE[1]]))
        data_deposit = window.GUI.slider_float('Data deposit', data_deposit, 0.0, MAX_DEPOSIT)
        agent_deposit = window.GUI.slider_float('Agent deposit', agent_deposit, 0.0, 10.0 * MAX_DEPOSIT * DATA_TO_AGENTS_RATIO)
        deposit_attenuation = window.GUI.slider_float('Deposit attn', deposit_attenuation, 0.8, 0.999)
        trace_attenuation = window.GUI.slider_float('Trace attn', trace_attenuation, 0.8, 0.999)
        
        ###
        ### Rendering controls
        ###
        def field_reset():
            vis_field.fill(0.0)
            return 0, 0
        
        window.GUI.text("Renderer parameters:")
        new_max_bounces = window.GUI.slider_int('Max Bounces', max_bounces, 1, 64)
        if (new_max_bounces != max_bounces):
            max_bounces = new_max_bounces
            accumulate_frame, accumulation_count = field_reset()
        
        new_num_samples = window.GUI.slider_int('Num Samples', num_samples, 1, 12)
        if (new_num_samples != num_samples):
            num_samples = new_num_samples
            accumulate_frame, accumulation_count = field_reset()
            
        new_throughput_multiplier = window.GUI.slider_float('Exposure', throughput_multiplier, 0.0, 10.0)
        if (new_throughput_multiplier != throughput_multiplier):
            throughput_multiplier = new_throughput_multiplier
            accumulate_frame, accumulation_count = field_reset()
            
        new_trace_vis = window.GUI.slider_float('Trace Vis', trace_vis, 0.0, 1.0)
        if (new_trace_vis != trace_vis):
            trace_vis = new_trace_vis
            accumulate_frame, accumulation_count = field_reset()

        new_deposit_vis = window.GUI.slider_float('Deposit Vis', deposit_vis, 0.0, 1.0)
        if (new_deposit_vis != deposit_vis):
            deposit_vis = new_deposit_vis
            accumulate_frame, accumulation_count = field_reset()

        new_trace_max = window.GUI.slider_float('Trace Max', trace_max, -4.0, 4.0)
        if (new_trace_max != trace_max):
            trace_max = new_trace_max
            accumulate_frame, accumulation_count = field_reset()
            
        new_albedo = window.GUI.slider_float('Albedo', albedo, 0.0, 0.99)
        if (new_albedo != albedo):
            albedo = new_albedo
            accumulate_frame, accumulation_count = field_reset()
            sigma_a, sigma_s = update_albedo_constraints(albedo, sigma_a, sigma_s, sigma_t)

        new_sigma_e = window.GUI.slider_float('Emission', sigma_e, 0.0, 100.0)
        if (new_sigma_e != sigma_e):
            sigma_e = new_sigma_e
            accumulate_frame, accumulation_count = field_reset()

        new_sigma_t = window.GUI.slider_float('Extinction', sigma_t, 0.0, 5.0)
        if (new_sigma_t != sigma_t):
            sigma_t = new_sigma_t
            accumulate_frame, accumulation_count = field_reset()
            sigma_a, sigma_s = update_albedo_constraints(albedo, sigma_a, sigma_s, sigma_t)

        new_scattering_anisotropy = window.GUI.slider_float('Scattering Anisotropy', scattering_anisotropy, 0.1, 0.98)
        if (new_scattering_anisotropy != scattering_anisotropy):
            scattering_anisotropy = new_scattering_anisotropy
            accumulate_frame, accumulation_count = field_reset()

        # new_ambient_trace = window.GUI.slider_float('Ambient Trace', ambient_trace, 0.0, 1.0)
        # if (new_ambient_trace != ambient_trace):
        #     ambient_trace = new_ambient_trace
        #     accumulate_frame = False
        #     accumulation_count = 0

        new_use_ray_marcher = window.GUI.checkbox("Use Ray Marcher", use_ray_marcher)
        if (new_use_ray_marcher != use_ray_marcher):
            use_ray_marcher = new_use_ray_marcher
            field_reset()

        new_ray_marcher_steps = window.GUI.slider_int('# Steps', ray_marcher_steps, 1, 400)
        if (new_ray_marcher_steps != ray_marcher_steps):
            ray_marcher_steps = new_ray_marcher_steps
            field_reset()

        new_debug_mode = window.GUI.checkbox("Debug Mode", use_debug_mode)
        if (new_debug_mode != use_debug_mode):
            use_debug_mode = new_debug_mode
            field_reset()

        ###
        ### Agent controls
        ###
        window.GUI.text("Distance distribution:")
        if window.GUI.checkbox("Constant", distance_sampling_distribution == EnumDistanceSamplingDistribution.CONSTANT):
            distance_sampling_distribution = EnumDistanceSamplingDistribution.CONSTANT
        if window.GUI.checkbox("Exponential", distance_sampling_distribution == EnumDistanceSamplingDistribution.EXPONENTIAL):
            distance_sampling_distribution = EnumDistanceSamplingDistribution.EXPONENTIAL
        if window.GUI.checkbox("Maxwell-Boltzmann", distance_sampling_distribution == EnumDistanceSamplingDistribution.MAXWELL_BOLTZMANN):
            distance_sampling_distribution = EnumDistanceSamplingDistribution.MAXWELL_BOLTZMANN

        window.GUI.text("Directional distribution:")
        if window.GUI.checkbox("Discrete", directional_sampling_distribution == EnumDirectionalSamplingDistribution.DISCRETE):
            directional_sampling_distribution = EnumDirectionalSamplingDistribution.DISCRETE
        if window.GUI.checkbox("Cone", directional_sampling_distribution == EnumDirectionalSamplingDistribution.CONE):
            directional_sampling_distribution = EnumDirectionalSamplingDistribution.CONE

        window.GUI.text("Directional mutation:")
        if window.GUI.checkbox("Deterministic", directional_mutation_type == EnumDirectionalMutationType.DETERMINISTIC):
            directional_mutation_type = EnumDirectionalMutationType.DETERMINISTIC
        if window.GUI.checkbox("Stochastic", directional_mutation_type == EnumDirectionalMutationType.PROBABILISTIC):
            directional_mutation_type = EnumDirectionalMutationType.PROBABILISTIC

        window.GUI.text("Deposit fetching:")
        if window.GUI.checkbox("Nearest neighbor", deposit_fetching_strategy == EnumDepositFetchingStrategy.NN):
            deposit_fetching_strategy = EnumDepositFetchingStrategy.NN
        if window.GUI.checkbox("Noise-perturbed NN", deposit_fetching_strategy == EnumDepositFetchingStrategy.NN_PERTURBED):
            deposit_fetching_strategy = EnumDepositFetchingStrategy.NN_PERTURBED

        window.GUI.text("Agent boundary handling:")
        if window.GUI.checkbox("Wrap around", agent_boundary_handling == EnumAgentBoundaryHandling.WRAP):
            agent_boundary_handling = EnumAgentBoundaryHandling.WRAP
        if window.GUI.checkbox("Reinitialize center", agent_boundary_handling == EnumAgentBoundaryHandling.REINIT_CENTER):
            agent_boundary_handling = EnumAgentBoundaryHandling.REINIT_CENTER
        if window.GUI.checkbox("Reinitialize randomly", agent_boundary_handling == EnumAgentBoundaryHandling.REINIT_RANDOMLY):
            agent_boundary_handling = EnumAgentBoundaryHandling.REINIT_RANDOMLY

        window.GUI.text("Misc controls:")
        do_simulate = window.GUI.checkbox("Run simulation", do_simulate)
        do_render = window.GUI.checkbox('Run renderer', do_render)
        if not do_render:
            vis_field.fill(0.0)
        do_export = do_export | window.GUI.button('Export fit')
        do_screenshot = do_screenshot | window.GUI.button('Screenshot')
        do_quit = do_quit | window.GUI.button('Quit')
        window.GUI.end()

        ## Help window
        ## Do not exceed prescribed line length of 120 characters, there is no text wrapping in Taichi GUI for now
        window.GUI.begin('Help', 0.35 * 1024.0 / FLOAT_CPU(VIS_RESOLUTION[0]), 0.01, 0.6, 0.30 * 1024.0 / FLOAT_CPU(VIS_RESOLUTION[1]))
        window.GUI.text("Welcome to PolyPhy 3D GUI variant written by researchers at UCSC/OSPO with the help of numerous external contributors\n(https://github.com/PolyPhyHub). PolyPhy implements MCPM, an agent-based, stochastic, pattern forming algorithm designed\nby Elek et al, inspired by Physarum polycephalum slime mold. Below is a quick reference guide explaining the parameters\nand features available in the interface. The reference as well as other panels can be hidden using the arrow button, moved,\nand rescaled.")
        window.GUI.text("")
        window.GUI.text("PARAMETERS")
        window.GUI.text("Sensing dist: average distance in world units at which agents probe the deposit")
        window.GUI.text("Sensing angle: angle in radians within which agents probe deposit (left and right concentric to movement direction)")
        window.GUI.text("Sampling expo: sampling sharpness (or 'acuteness' or 'temperature') which tunes the directional mutation behavior")
        window.GUI.text("Step size: average size of the step in world units which agents make in each iteration")
        window.GUI.text("Data deposit: amount of marker 'deposit' that *data* emit at every iteration")
        window.GUI.text("Agent deposit: amount of marker 'deposit' that *agents* emit at every iteration")
        window.GUI.text("Deposit attn: attenuation (or 'decay') rate of the diffusing combined agent+data deposit field")
        window.GUI.text("Trace attn: attenuation (or 'decay') of the non-diffusing agent trace field")
        window.GUI.text("Deposit vis: visualization intensity of the green deposit field (logarithmic)")
        window.GUI.text("Trace vis: visualization intensity of the red trace field (logarithmic)")
        window.GUI.text("")
        window.GUI.text("OPTIONS")
        window.GUI.text("Distance distribution: strategy for sampling the sensing and movement distances")
        window.GUI.text("Directional distribution: strategy for sampling the sensing and movement directions")
        window.GUI.text("Directional mutation: strategy for selecting the new movement direction")
        window.GUI.text("Deposit fetching: access behavior when sampling the deposit field")
        window.GUI.text("Agent boundary handling: what do agents do if they reach the boundary of the simulation domain")
        window.GUI.text("")
        window.GUI.text("NAVIGATION")
        window.GUI.text("Right Mouse: rotate camera")
        window.GUI.text("Middle Mouse: zoom camera")
        window.GUI.text("Up/Down Arrow: zoom camera")
        window.GUI.text("")
        window.GUI.text("VISUALIZATION")
        window.GUI.text("Renders 2 types of information superimposed on top of each other: *green* deposit field and *red-purple* trace field.")
        window.GUI.text("Yellow-white signifies areas where deposit and trace overlap (relative intensities are controlled by the T/D vis params)")
        window.GUI.text("Screenshots can be saved in the /capture folder.")
        window.GUI.text("")
        window.GUI.text("DATA")
        window.GUI.text("Input data are loaded from the specified folder in /data. Currently the CSV format is supported.")
        window.GUI.text("Reconstruction data are exported to /data/fits using the Export fit button.")
        window.GUI.end()

    ## Main simulation sequence
    if do_simulate:
        data_step(data_deposit, current_deposit_index)
        agent_step(\
            sense_distance,\
            sense_angle,\
            STEERING_RATE,\
            sampling_exponent,\
            step_size,\
            agent_deposit,\
            current_deposit_index,\
            distance_sampling_distribution,\
            directional_sampling_distribution,\
            directional_mutation_type,\
            deposit_fetching_strategy,\
            agent_boundary_handling)
        deposit_relaxation_step(deposit_attenuation, current_deposit_index)
        trace_relaxation_step(trace_attenuation)
        current_deposit_index = 1 - current_deposit_index

    ## Render visualization
    if do_render:
        if use_ray_marcher == True:
            render_visualization_raymarched(throughput_multiplier, deposit_vis, trace_vis, camera_distance, camera_polar, camera_azimuth, ray_marcher_steps, current_deposit_index)
        else:
            render_visualization_volumetric_pathtraced(camera_distance, camera_polar, camera_azimuth, accumulate_frame, accumulation_count, throughput_multiplier / 10, num_samples, max_bounces, deposit_vis / 10, trace_vis / 10, scattering_anisotropy, sigma_e, sigma_a, sigma_s, sigma_t, trace_max, use_debug_mode)
        canvas.set_image(vis_field)

    if do_screenshot:
        window.write_image(ROOT + 'capture/screenshot_' + stamp() + '.png') ## Must appear before window.show() call
    window.show()
    if do_export:
        store_fit()
    if do_quit:
        break

window.destroy()
