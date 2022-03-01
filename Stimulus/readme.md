# Stimulus 

The primary file in this folder, SeqRFExp.m, was used to generate visual stimuli in our experiments. Most of the code simply creates relevant variables, while the SequenceStim.frag.txt and SequenceStim.vert.txt use OpenGL code to communicate with the GPU for the actual stimulus generation. This was used in conjunction with Psychtoolbox3 and an interface to a DAQ USB-1208FS-Plus that communicated synchronization signals to the OpenEphys recording equipment that we used to do electrophysiology. 

