# ...
"""This module contains the operation functions for this project.

Functions defined in this module are called from the run.py module."""
from __future__ import print_function, division, absolute_import

"""Definition of constants relevant for this project."""
PARTICLE_VOLUME = 1

from flow import FlowProject
import json

@FlowProject.label
def done(job):
    return job.document.get('completed',False)

@FlowProject.label
def fv_done(job):
    return False #(job.document.get('free_volumes') != None)


@FlowProject.operation
@FlowProject.post(done)
def frenkel_ladd(job):
    import hoomd
    from hoomd import hpmc
    from hoomd import deprecated
    import sys
    sys.path.append('../utils/')
    import shapetools
    from garnett.reader import GSDHOOMDFileReader
    #import initialization_structures as structs
    from euclid.FreudShape import ConvexPolyhedron
    import numpy as np
    import os
    import json
    import time

    hoomd.context.initialize("--mode=cpu")

    with job:
        sp = job.statepoint()

        truncation = sp['truncation']
        #structure = sp['structure']
        #pf = sp['pf']
        polyhedra_type = "A"
#        polyhedra_type = "Polyhedra"
        mverts = 8 if truncation == 1 else 12

        # Set up variables
        c_val = (2*((-truncation)+1)+1) # Mapping 0-1 truncation onto the 3-1 range of shapetools
        log_quantities = ['time', 'hpmc_sweeps', 'hpmc_translate_acceptance', 'hpmc_rotate_acceptance', 'hpmc_d', 'hpmc_a', 'hpmc_overlap_count', 'volume', 'hpmc_move_ratio', 'lattice_energy_pp_avg', 'lattice_energy_pp_sigma', 'lattice_translational_spring_constant', 'lattice_rotational_spring_constant', 'lattice_num_samples']

        # Construct the shape
        particle_vertices = shapetools.setVolumeVerts(a = 1, b = 2, c = c_val, Vol = PARTICLE_VOLUME)
        freud_shape = ConvexPolyhedron(points = particle_vertices)
        particle_radius = freud_shape.getCircumsphereRadius()

        #uc = structs.get_unit_cell(structure, truncation, type_name = polyhedra_type)

        # Determine how much to replicate
        #desired_N = 2**9
        fname_base = 'calc_FL'
        log_file = fname_base + '.log'
        restart_file = fname_base + '_restart.gsd'
        output_file = fname_base + '.gsd'
        pos_file = fname_base + '.pos'

        snap = hoomd.data.gsd_snapshot("/scratch/sglotzer_fluxoe/plawton/transfercode/frenkel-ladd/"+sp['structure']+".gsd")
        #n = int(round((desired_N/snap.particles.N)**(1./3.)))
        system = hoomd.init.read_snapshot(snap) if not job.isfile(restart_file) else hoomd.init.read_gsd(restart_file)#hoomd.init.create_lattice(uc, n)
        num_particles = len(system.particles)
        job.document['N']=num_particles
        job.document['random_seed']=job.document.get('random_seed', np.random.randint(1,1e5))
        seed = job.document['random_seed']

        # Resize the system to the approximate densest packing
        vol_init = system.box.get_volume()
        pf_init = num_particles*PARTICLE_VOLUME/vol_init

        overlap_log = hoomd.analyze.log(filename=None, quantities=['hpmc_overlap_count'], period=1)
        overlap_mc = hpmc.integrate.convex_polyhedron(seed=seed, max_verts=mverts)
        overlap_mc.set_params(d=0.0, a=0.0)
        overlap_mc.shape_param.set(polyhedra_type, vertices = particle_vertices)
        hoomd.run(1)
        radius_ratio = 0.05
        depletant_radius = particle_radius * radius_ratio
        depletant_vol = (4/3) * np.pi * depletant_radius**3
        scale = 1 + (depletant_vol/vol_init)**(1/3)
        while overlap_log.query('hpmc_overlap_count') > 0:
            hoomd.update.box_resize(Lx=system.box.Lx*scale,Ly=system.box.Ly*scale,Lz=system.box.Lz*scale, period=None)
            hoomd.run(1)
        job.document['densest_pf'] = (num_particles*PARTICLE_VOLUME)/(system.box.get_volume())

        mc = hpmc.integrate.convex_polyhedron(seed=seed, max_verts = mverts)
        mc.shape_param.set(polyhedra_type, vertices = particle_vertices)
        mc.set_params(d = 0.1, a = 0.1)
        snap = system.take_snapshot()
        fl = hpmc.field.frenkel_ladd_energy(mc = mc, ln_gamma = 0.0, q_factor = 10.0, r0 = snap.particles.position, q0 = snap.particles.orientation, drift_period = 1000)

        hoomd.dump.gsd(filename = restart_file, group = hoomd.group.all(), period = 10000, phase = 0, truncate = True)
        hoomd.dump.gsd(filename = output_file, group = hoomd.group.all(), phase = 0, period = 10000, overwrite = False)
#        pos = deprecated.dump.pos(filename = pos_file, phase = 0, period = 10000)
 #       mc.setup_pos_writer(pos)

        particle_tunables=['d','a']
        max_part_moves=[particle_radius,0.5]
        particle_tuner = hpmc.util.tune(obj = mc, tunables = particle_tunables, max_val = max_part_moves, gamma = 2.0, target = 0.3)
        N_target_moves = int(2e5)
        num_tuning_steps = 20
        tuner_period = int(np.floor(N_target_moves/num_tuning_steps)) + 1
        fl_period = int(2e5)
        fl_eq_buffer = int(4e4)
        hoomd.run(1)
        log = hoomd.analyze.log(filename = log_file, header_prefix = '#', quantities = log_quantities, period = fl_period+fl_eq_buffer+num_tuning_steps*tuner_period, overwrite = False)

        #sweep of spring constants
        try:
            ln_gammas = np.linspace(15,-5,21)
            start = ln_gammas.index(job.document('gamma_completed'))+1 if 'gamma_completed' in job.document else 0
            for ln_gam in ln_gammas[start:]:
                #tune the step size
                fl.set_params(ln_gamma = ln_gam)
                for i in range(num_tuning_steps):
                    #hoomd.run(tuner_period, quiet = True)
                    hoomd.run(tuner_period)
                    particle_tuner.update()
                hoomd.run(fl_eq_buffer)
                fl.reset_statistics()
                hoomd.run(fl_period)
            job.document['steps_completed'] = hoomd.get_step()
            job.document['gamma_completed']=ln_gam
        except hoomd.WalltimeLimitReached:
            # Since I've already run a bunch of these jobs without any thought to restartability,
            # for now I'm just going to write this such that I will restart the run if something
            # terminates the run. Since the runs are relatively short this is not a big loss.
            job.document['steps_completed'] = hoomd.get_step()
            job.document['completed'] = False

        job.document['completed'] = True

@FlowProject.operation
@FlowProject.post(fv_done)
def compute_fv(job):
    import hoomd
    from hoomd import hpmc
    from hoomd import deprecated
    hoomd.context.initialize("--mode=cpu")
    import sys
    sys.path.append('../utils/')
    from utils import shapetools
    from euclid.FreudShape import ConvexPolyhedron
    from garnett.reader import GSDHOOMDFileReader
    import numpy as np
    import os
    import json

    with job:
        sp = job.statepoint()

        truncation = sp['truncation']
        structure = sp['structure']
        pf = sp['pf']
        polyhedra_type = "A" #"Polyhedra"
        depletant_type = "Depletant"
        mverts = 8 if truncation == 1 else 12

        # Set up variables
        c_val = (2*((-truncation)+1)+1) # Mapping 0-1 truncation onto the 3-1 range of shapetools

        # Construct the shape
        particle_vertices = shapetools.setVolumeVerts(a = 1, b = 2, c = c_val, Vol = PARTICLE_VOLUME)
        freud_shape = ConvexPolyhedron(points = particle_vertices)
        particle_radius = freud_shape.getCircumsphereRadius()

        fname_base = 'calc_fv'
        log_file = fname_base + '.log'
        restart_file = fname_base + '_restart.gsd'

        snap = hoomd.data.gsd_snapshot("/scratch/sglotzer_fluxoe/plawton/transfercode/frenkel-ladd/"+sp['structure']+".gsd")
        system = hoomd.init.read_snapshot(snap) if not job.isfile(restart_file) else hoomd.init.read_gsd(restart_file)#hoomd.init.create_lattice(uc, n)
        num_particles = len(system.particles)
        job.document['N']=num_particles

        # Resize the system to the approximate densest packing
        vol_init = system.box.get_volume()
        pf_init = num_particles*PARTICLE_VOLUME/vol_init

        job.document['random_seed']=job.document.get('random_seed', np.random.randint(1,1e5))
        seed = job.document['random_seed']

        if pf > pf_init :
        # First check that the new packing fraction is even feasible (i.e. it must be < the initial volume, otherwise we're trying a state point that can't be compressed to.
            job.document['valid_pf'] = False
            raise ValueError('This packing fraction is higher than the densest packing determined!')
        else:
            job.document['valid_pf'] = True
            vol = (num_particles*PARTICLE_VOLUME)/pf
            scale = (vol/vol_init)**(1./3)
            hoomd.update.box_resize(Lx = system.box.Lx*scale, Ly = system.box.Ly*scale, Lz = system.box.Lz*scale, period = None)
        #seed = num_particles*sum(ord(c) for c in structure)

        system.particles.types.add(depletant_type)
        mc = hpmc.integrate.convex_spheropolyhedron(seed = int(seed), implicit=True, max_verts = mverts)
        mc.shape_param.set(polyhedra_type, vertices = particle_vertices, sweep_radius = 0) # Set R to 0 for a perfect polyhedron (no rounding of the edges)
        mc.set_params(depletant_type = depletant_type)
        mc.set_params(d = {polyhedra_type: 0, depletant_type: 0}, a = {polyhedra_type: 0, depletant_type: 0})
        mc.set_params(nR = 0) # Don't actually include depletants
        free_vol = hpmc.compute.free_volume(mc = mc, seed = 4980293, nsample = 500000, test_type = depletant_type)

        hoomd.dump.gsd(filename = restart_file, group = hoomd.group.all(), period = 10000, phase = 0, truncate = True)

        log_quantities = ['time', 'hpmc_sweep', 'hpmc_translate_acceptance', 'hpmc_rotate_acceptance', 'hpmc_d', 'hpmc_a', 'hpmc_overlap_count', 'volume', 'hpmc_move_ratio', 'hpmc_free_volume']
        log = hoomd.analyze.log(filename = log_file, header_prefix = '#', quantities = log_quantities, period = 1, overwrite = True)

	# Compute any properties that are specific to each depletion
        radius_ratios = [0.03, 0.04, 0.05, 0.047140452079103182, 0.06, 0.07, 0.08, 0.09] + np.linspace(0.1,0.4, 15).tolist()
        job.document['radius_ratios'] = radius_ratios

        # Get existing free volumes if they are there.
        try:
            avg_free_vols = job.document['free_volumes']
        except KeyError:
            avg_free_vols = []
        for q in radius_ratios:
            if not str(q) in avg_free_vols:
                free_vols = []
                depletant_radius = particle_radius*q
                mc.shape_param.set(depletant_type, vertices = [], sweep_radius = depletant_radius)
                hoomd.run(1e2, callback_period = 1, callback = lambda step: free_vols.append(log.query('hpmc_free_volume')))
                avg_free_vol = sum(free_vols) / len(free_vols)
                avg_free_vols.append(avg_free_vol)
        job.document['free_volumes'] = avg_free_vols

#        for q in radius_ratios:
#            # Avoid recomputing. Because it's JSONified, have to check the string version
#            if not str(q) in avg_free_vols:
#                free_vols = []
#                depletant_radius = particle_radius*q
#                mc.shape_param.set(depletant_type, vertices = [], sweep_radius = depletant_radius)
#                hoomd.run(1e2, callback_period = 1, callback = lambda step: free_vols.append(log.query('hpmc_free_volume')))
#                avg_free_vol = sum(free_vols) / len(free_vols)
#                avg_free_vols[q] = avg_free_vol

        #job.document['free_volume'] = avg_free_vol

if __name__ == '__main__':
    FlowProject().main()
