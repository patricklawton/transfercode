"""This module contains the operation functions for this project."""
from __future__ import print_function, division, absolute_import

import signac
from flow import FlowProject, directives, cmd
try:
    # Where available (e.g. flux)
    import glotzer_environments
except:
    # Otherwise just use flow environments
    import flow.environments  # uncomment to use default environments
import contextlib

# Constants
NUM_STEPS = 10e8

class AssemblyProject(FlowProject):
    pass

@AssemblyProject.label()
def compressed(job):
    return job.document.get('compressed') is not None and job.isfile('compress_fluid.gsd')

@AssemblyProject.label()
def tuned(job):
    return job.document.get('tuned') is not None

@AssemblyProject.label()
def finished(job):
    """Use a timestep based solution instead for easier checking"""
    return job.document.get('finished') is not None

@AssemblyProject.operation
@AssemblyProject.pre.never  # For now, only sedimentation
@AssemblyProject.post.true('compressed')
def compress_fluid(job):
    # Load packages as necessary
    import hoomd
    from hoomd import hpmc
    from utils.hoomd_utils import _redirect_hoomd_log
    from utils.Compressor import Compressor
    from utils import shapetools
    from euclid.FreudShape import ConvexPolyhedron
    import datetime

    if hoomd.context.exec_conf is None:
        hoomd.context.initialize('--mode=cpu')
    with job:
        with _redirect_hoomd_log(job):
            with hoomd.context.SimulationContext():
                polyhedra_type = "Polyhedra"
                fname_base = 'compress_fluid'
                log_file = fname_base + '.log'
                restart_file = fname_base + '_restart.gsd'
                output_file = fname_base + '.gsd'

                # Calculate the c parameter for shapetools based on the
                # truncation parameter passed in. We are mapping the range 0-1
                # onto the range 3-1
                c_val = (2 * ((-job.sp.truncation) + 1) + 1)
                particle_vertices = shapetools.unitVolumeVerts(
                    a=1, b=2, c=c_val)
                freud_shape = ConvexPolyhedron(points=particle_vertices)
                particle_radius = freud_shape.getCircumsphereRadius()

                # Just creating a simple cubic lattice # is fine here.
                if job.isfile(restart_file):
                    system = hoomd.init.read_gsd(restart_file)
                else:
                    system = hoomd.init.create_lattice(hoomd.lattice.sc(
                        particle_radius * 2, type_name=polyhedra_type), n=10)

                mc = hpmc.integrate.convex_polyhedron(seed=job.sp.run_num)
                mc.shape_param.set(polyhedra_type, vertices=particle_vertices)
                mc.set_params(d=0.1, a=0.1)

                # Dump files
                gsd = hoomd.dump.gsd(
                    filename=output_file, group=hoomd.group.all(), period=10000, phase=0)
                gsd.dump_state(mc)
                restart = hoomd.dump.gsd(filename=restart_file, group=hoomd.group.all(),
                        period=10000, phase=0, truncate = True)
                restart.dump_state(mc)
                log_quantities = ['time', 'hpmc_overlap_count', 'hpmc_translate_acceptance',
                                  'hpmc_rotate_acceptance', 'hpmc_d', 'hpmc_a', 'volume']
                log = hoomd.analyze.log(
                    filename=log_file, header_prefix='#', quantities=log_quantities, period=500)

                # Define a tuner
                mc_tune = hpmc.util.tune(mc, tunables=['d', 'a'], max_val=[
                                         particle_radius, 0.5], gamma=1, target=0.3)

                # Try to resolve overlaps, and stop if we can't
                hoomd.run(1)
                for i in range(5):
                    if mc.count_overlaps() == 0:
                        break
                    hoomd.run(100)
                if mc.count_overlaps() > 0:
                    raise ValueError('There are {} overlaps in your system'.format(
                        mc.count_overlaps()))

                try:
                    comp = Compressor(system, mc)
                    comp.set_tuner(mc_tune)
                    system = comp.compress_to_pf(job.sp.packing_fraction)
                    if hoomd.comm.get_rank() == 0:
                        metadata = job.document.get('hoomd_meta', dict())
                        metadata[str(datetime.datetime.now().timestamp())] = hoomd.meta.dump_metadata()
                        job.document['hoomd_meta'] = metadata
                        job.document['steps_compressed'] = hoomd.get_step()
                        job.document['compressed'] = True
                except RuntimeError:
                    gsd.write_restart()


@AssemblyProject.operation
@AssemblyProject.pre.never  # Disable auto running since we should always submit the mpi version
@AssemblyProject.post.true('finished')
def assemble(job):
    # Load packages as necessary
    import hoomd
    from hoomd import hpmc
    from utils.hoomd_utils import _redirect_hoomd_log
    from utils.Compressor import Compressor
    from utils import shapetools
    import math
    from euclid.FreudShape import ConvexPolyhedron
    import datetime
    import numpy as np

    if hoomd.context.exec_conf is None:
        hoomd.context.initialize('--mode=cpu')
    with job:
        with _redirect_hoomd_log(job):
            with hoomd.context.SimulationContext():
                polyhedra_type = "Polyhedra"
                depletant_type = 'Depletant'

                # File name setup
                fname_base = 'assemble'
                init_file = 'compress_fluid.gsd'
                log_file = fname_base + '.log'
                restart_file = fname_base + '_restart.gsd'
                output_file = fname_base + '.gsd'

                # Calculate the c parameter for shapetools based on the
                # truncation parameter passed in. We are mapping the range 0-1
                # onto the range 3-1
                c_val = (2 * ((-job.sp.truncation) + 1) + 1)
                particle_vertices = shapetools.unitVolumeVerts(
                    a=1, b=2, c=c_val)
                freud_shape = ConvexPolyhedron(points=particle_vertices)
                particle_radius = freud_shape.getCircumsphereRadius()

                if job.isfile(restart_file):
                    system = hoomd.init.read_gsd(filename=restart_file)
                else:
                    with hoomd.context.SimulationContext() as c:
                        temp_system = hoomd.init.read_gsd(init_file, frame=-1)
                        snap_init = temp_system.take_snapshot()
                    system = hoomd.init.read_snapshot(snap_init)
                    system.particles.types.add(depletant_type)

                # Now set up the integrator
                depletant_radius = particle_radius * job.sp.radius_ratio
                depletant_volume = (4.0 / 3) * math.pi * \
                    (depletant_radius**3)
                depletant_number_density = job.sp.packing_fraction_depletant / depletant_volume

                mc = hpmc.integrate.convex_spheropolyhedron(
                    seed=job.sp.run_num, implicit=True, depletant_mode='overlap_regions')
                mc.shape_param.set(
                    polyhedra_type, vertices=particle_vertices, sweep_radius=0)
                mc.set_params(nR=depletant_number_density)
                mc.set_params(depletant_type=depletant_type)
                mc.set_params(d={polyhedra_type: 0.01, depletant_type: 0.1}, a={
                              polyhedra_type: 0.01, depletant_type: 0.1})
                mc.shape_param.set(depletant_type, vertices=[],
                                   sweep_radius=depletant_radius)
                free_vol = hpmc.compute.free_volume(
                    mc=mc, seed=123, nsample=500000, test_type=depletant_type)
                log_quantities = ['time', 'hpmc_sweep', 'hpmc_overlap_count',
                                  'hpmc_translate_acceptance',
                                  'hpmc_rotate_acceptance', 'hpmc_d', 'hpmc_a',
                                  'volume', 'hpmc_free_volume',
                                  'hpmc_fugacity']

                # Dump files
                gsd = hoomd.dump.gsd(
                    filename=output_file, group=hoomd.group.all(), period=10000, phase=0)
                gsd.dump_state(mc)
                restart = hoomd.dump.gsd(filename=restart_file, group=hoomd.group.all(),
                        period=10e3, phase=0, truncate=True)
                restart.dump_state(mc)
                log = hoomd.analyze.log(
                    filename=log_file, header_prefix='#', quantities=log_quantities, period=500)

                # Tune if this is the first time
                tuner_period = 500
                tuning_steps = 10
                if job.document.get('tuned') is None:
                    # Allow things to run by first randomizing the system and
                    # then tuning the step size
                    mc_tune = hpmc.util.tune(mc, tunables=['d', 'a'], max_val=[
                                             particle_radius, 0.5], gamma=1, target=0.3)
                    for i in range(tuning_steps):
                        mc_tune.update()
                        hoomd.run(tuner_period)

                    if hoomd.comm.get_rank() == 0:
                        job.document['tuned'] = True
                        job.document['d_tuned'] = mc.get_d()
                        job.document['a_tuned'] = mc.get_a()
                else:
                    mc.set_params(d=job.document.get('d_tuned'),
                                  a=job.document.get('a_tuned'))
                try:
                    runupto_val = NUM_STEPS + tuner_period * tuning_steps
                    hoomd.run_upto(runupto_val)
                    # Dump final output
                    if hoomd.comm.get_rank() == 0:
                        metadata = job.document.get('hoomd_meta', dict())
                        metadata[str(datetime.datetime.now().timestamp())] = cast_json(hoomd.meta.dump_metadata())
                        job.document['hoomd_meta'] = metadata
                        job.document['finished'] = True
                except hoomd.WalltimeLimitReached:
                    if hoomd.comm.get_rank() == 0:
                        job.document['num_steps'] = hoomd.get_step()


@AssemblyProject.operation
# @AssemblyProject.pre.never  # Disable auto running since we should always submit the mpi version
# @AssemblyProject.pre(lambda job: job.sp.packing_fraction_depletant == 0.05)
@AssemblyProject.post.true('sedimented')
@directives(nranks=12)
def sediment(job):
    """This operation simulates a sedimentation experiment by using an elongated box in the z-dimension and adding an effective gravitational potential (in the absence of depletion)."""
    import hoomd
    from hoomd import hpmc, jit
    from utils.hoomd_utils import _redirect_hoomd_log
    from utils.Compressor import Compressor
    from utils import shapetools
    from euclid.FreudShape import ConvexPolyhedron
    import datetime
    import numpy as np

    if hoomd.context.exec_conf is None:
        hoomd.context.initialize('--mode=cpu')
    with job:
        #with _redirect_hoomd_log(job):
        if True:
            with hoomd.context.SimulationContext():
                polyhedra_type = "Polyhedra"
                fname_base = 'sediment'
                log_file = fname_base + '.log'
                restart_file = fname_base + '_restart.gsd'
                output_file = fname_base + '.gsd'

                # Calculate the c parameter for shapetools based on the
                # truncation parameter passed in. We are mapping the range 0-1
                # onto the range 3-1
                c_val = (2 * ((-job.sp.truncation) + 1) + 1)
                particle_vertices = shapetools.unitVolumeVerts(
                    a=1, b=2, c=c_val)
                freud_shape = ConvexPolyhedron(points=particle_vertices)
                particle_radius = freud_shape.getCircumsphereRadius()

                # Just creating a simple cubic lattice # is fine here.
                if job.isfile(restart_file):
                    system = hoomd.init.read_gsd(restart_file)
                else:
                    xydim = 12
                    system = hoomd.init.create_lattice(hoomd.lattice.sc(
                        particle_radius * 2, type_name=polyhedra_type), n=[xydim,xydim,2])# was n=16

                mc = hpmc.integrate.convex_polyhedron(seed=job.sp.run_num)
                mc.shape_param.set(polyhedra_type, vertices=particle_vertices)
                mc.set_params(d=0.1, a=0.1)

                # Dump files
                gsd = hoomd.dump.gsd(
                    filename=output_file, group=hoomd.group.all(), period=1000, phase=0, overwrite=False)
                gsd.dump_state(mc)
                restart = hoomd.dump.gsd(filename=restart_file, group=hoomd.group.all(),
                        period=10000, phase=0, truncate = True)
                restart.dump_state(mc)
                log_quantities = ['time', 'hpmc_overlap_count', 'hpmc_translate_acceptance',
                                  'hpmc_rotate_acceptance', 'hpmc_d', 'hpmc_a', 'volume',
                                  'external_field_jit']
                log = hoomd.analyze.log(
                    filename=log_file, header_prefix='#', quantities=log_quantities, period=500, overwrite=False)

                # Define a tuner
                mc_tune = hpmc.util.tune(mc, tunables=['d', 'a'], max_val=[
                                         particle_radius, 0.5], gamma=1, target=0.3)

                # Tune if this is the first time
                tuner_period = 500
                tuning_steps = 4
                if job.document.get('sediment_tuned') is None:
                    # Allow things to run by first randomizing the system and
                    # then tuning the step size. We tune before box resizing so
                    # that the move sizes will be representative of the
                    # diffusivity in the dense system.
                    mc_tune = hpmc.util.tune(mc, tunables=['d', 'a'], max_val=[
                                             particle_radius, 0.5], gamma=1, target=0.3)
                    for i in range(tuning_steps):
                        mc_tune.update()
                        hoomd.run(tuner_period)

                    if hoomd.comm.get_rank() == 0:
                        job.document['sediment_tuned'] = True
                        job.document['sediment_d_tuned'] = mc.get_d()
                        job.document['sediment_a_tuned'] = mc.get_a()
                else:
                    mc.set_params(d=job.document.get('sediment_d_tuned'),
                                  a=job.document.get('sediment_a_tuned'))

                # Expand system, add walls, and add gravity
                hoomd.update.box_resize(Lx=system.box.Lx, Ly = system.box.Ly,
                                        Lz=(2*system.box.Lx), scale_particles=False,
                                        period=None)
                wall = hpmc.field.wall(mc)
                wall.add_plane_wall([0, 0, 1], [0, 0, -system.box.Lz/2])
                # The gravitational force should be scaled such that it's on
                # the order of a kT per particle to get the right physics.
                gravity_field = hoomd.jit.external.user(mc=mc, code="return " + str(job.sp.g_val) + "*r_i.z + box.getL().z/2;")
                comp = hpmc.field.external_field_composite(mc, [wall, gravity_field])

                try:
                    runupto_val = 900000 #NUM_STEPS + tuner_period * tuning_steps
                    hoomd.run_upto(runupto_val)
                    # Dump final output
                    if hoomd.comm.get_rank() == 0:
                        #metadata = job.document.get('hoomd_meta', dict())
                        #metadata[str(datetime.datetime.now().timestamp())] = hoomd.meta.dump_metadata()
                        #job.document['hoomd_meta'] = metadata
                        job.document['sedimented'] = True
                except hoomd.WalltimeLimitReached:
                    if hoomd.comm.get_rank() == 0:
                        job.document['num_steps_sedimented'] = hoomd.get_step()



if __name__ == '__main__':
    AssemblyProject().main()
