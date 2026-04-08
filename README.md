# scotty2ermes
Links SCOTTY and ERMES for beam-tracing and full-wave simulations + analysis for DBS

examples.py contains 5 example use cases

'helpers' contains additional scripts that aid in transitioning between SCOTTY and ERMES

To use these with SCOTTY, you would first run your SCOTTY simulation to get an output .h5 file
Afterwhich, you run get_ERMES_params() using your SCOTTY .h5 file to get the necessary full-wave parameters.
This could probably work for other full-wave solvers like COMSOL, but I have yet to test this. 
Refer to ERMES 20.0+ documentation if you wan't to try your hand at it as the polarization vector and E-components are based off of this.

gen2D/3Dfullwavefile.py maps a SCOTTY/TORBEAM format ne.dat and topfile.json to what ERMES can use.
If you are using a different full-wave solver, you can take inspiration from this.

Once you run your ERMES simulation, the various analysis and plotting functions can be used to analyze the ERMES results beyond what is built into GiD.
Refer to examples.py for examples of how to do this.