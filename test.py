import hoomd
from hoomd import hpmc
from utils.hoomd_utils import _redirect_hoomd_log
from utils.Compressor import Compressor
from utils import shapetools
from euclid.FreudShape import ConvexPolyhedron
import datetime
import numpy as np

hoomd.context.initialize('--mode=cpu')

fname_base = 'sediment'
log_file = fname_base + '.log'
restart_file = fname_base + '_restart.gsd'
output_file = fname_base + '.gsd'


c_val = (2 * ((-0.55) + 1) + 1)
particle_vertices = shapetools.unitVolumeVerts(a=1, b=2, c=c_val)
freud_shape = ConvexPolyhedron(points=particle_vertices)
particle_radius = freud_shape.getCircumsphereRadius()

dim = 16
system = hoomd.init.create_lattice(hoomd.lattice.sc(particle_radius * 2, type_name="Polyhedra"), n=[dim,dim,1])

mc = hpmc.integrate.convex_polyhedron(seed=1) #does it really matter what I put here?
mc.shape_param.set("Polyhedra", vertices=particle_vertices)
mc.set_params(d=0.1, a=0.1)

# Dump files
gsd = hoomd.dump.gsd(filename=output_file, group=hoomd.group.all(), period=10000, phase=0, overwrite=False)
gsd.dump_state(mc)
#restart = hoomd.dump.gsd(filename=restart_file, group=hoomd.group.all(),period=10000, phase=0, truncate = True)
#restart.dump_state(mc)
#log_quantities = ['time', 'hpmc_overlap_count', 'hpmc_translate_acceptance','hpmc_rotate_acceptance', 'hpmc_d', 'hpmc_a', 'volume','external_field_jit']
#log = hoomd.analyze.log(filename=log_file, header_prefix='#', quantities=log_quantities, period=500, overwrite=False)

